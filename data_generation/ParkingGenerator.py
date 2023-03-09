import os
import torch
import random
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.transforms import euler_angles_to_matrix
from torch.utils.data.distributed import DistributedSampler

from .dataset import ParkingDataset, collate_fn
from .generator import Generator

from logger import get_logger
logger = get_logger("Parking generator logger")

import pdb

class ParkingGenerator(Generator):
    def __init__(self, cfg, rank, world_size, image_size=384, device='cpu'):
        super(ParkingGenerator, self).__init__(cfg, rank, world_size, device=device)
    
    def generate_set_multi(self, set_type, dataloader):
        assert set_type in ['train', 'validation', 'test'], "Set type must be one of train/val/test!"
        data_dir = os.path.join(self.cfg['DATA_DIR'], set_type)
        save_dir = os.path.join(self.cfg['SAVE_DIR'], set_type)
        image_counter = 0
        logger.info(f"Generating {set_type} dataset")
        
        # Populate images with rendered vehicles
        for images_batch, annotations_batch in tqdm(dataloader):
            batch_size = len(images_batch)
            
            # No matrices in the parking lot scenario
            matrix_info = [{'is_matrix': False, 'height': 0, 'width': 0} for _ in range(batch_size)]
            
            # Sample rendering parameters
            distance = self.cfg['RENDERING']['DISTANCE']
            elevation = self.cfg['RENDERING']['ELEVATION']
            azimuth = self.cfg['RENDERING']['AZIMUTH']
            sf_range = self.cfg['RENDERING']['SCALING_FACTORS_RANGE']
            int_range = self.cfg['RENDERING']['INTENSITIES_RANGE']
            distances = [distance for _ in range(batch_size)]
            elevations = [elevation for _ in range(batch_size)]
            azimuths = [azimuth for _ in range(batch_size)]
            lights_directions = torch.rand(batch_size, 3, device=self.device) * 2 - 1
            lights_directions[:, 1] = -1
            scaling_factors = torch.rand(batch_size, 1, device=self.device) * (sf_range[1] - sf_range[0]) + sf_range[0]
            intensities = torch.rand(batch_size, 1, device=self.device) * (int_range[1] - int_range[0]) + int_range[0]
            
            # Generate all possible scenarios for the given parking lot image
            for image, annotations, scaling_factor in zip(images_batch, annotations_batch, scaling_factors):
                meshes_list, offsets_list = self.populate_parking_lot(annotations, set_type, scaling_factor)
                assert len(meshes_list) == len(offsets_list)
            
                # Convert offsets to locations
                locations_list = []
                for offsets in offsets_list:
                    locations = torch.from_numpy(offsets).to(self.device)
                    locations = (1 + locations) * self.image_size / 2
                    locations_list.append(locations)
                anns_list = self.construct_annotations_files(locations_list)
        
                # Render images in batches
                batched_meshes_list = [meshes_list[x:x + self.cfg['BATCH_SIZE']] for x in range(0, len(meshes_list), self.cfg['BATCH_SIZE'])]
                batched_anns_list = [anns_list[x:x + self.cfg['BATCH_SIZE']] for x in range(0, len(anns_list), self.cfg['BATCH_SIZE'])]
                batch_counter = 0
                for meshes_joined, anns_joined in zip(batched_meshes_list, batched_anns_list):
                    logger.debug(f"Batch counter: {batch_counter}")
                    assert len(meshes_joined) == len(anns_joined)
                    bsz = len(meshes_joined)
                    
                    # Move the meshes to the device
                    meshes_joined = [mesh.to(self.device) for mesh in meshes_joined]
            
                    # Render the images
                    synthetic_images = self.renderer.render_batch(
                        meshes_joined, 
                        image.unsqueeze(0).repeat(bsz, 1, 1, 1), 
                        elevations[:bsz], 
                        azimuths[:bsz],
                        lights_directions[:bsz],
                        scaling_factors=scaling_factor.unsqueeze(0).repeat(bsz, 1),
                        intensities=intensities[:bsz],
                        distances=distances[:bsz],
                        image_size=self.image_size
                    )

                    # Save images and annotations
                    for k in range(len(anns_joined)):
                        # Image
                        synthetic_image = synthetic_images[k]
                        image_save_path = os.path.join(save_dir, "images", f"image_{self.rank}_{batch_counter}_{image_counter}_{k}.jpg")
                        self.tensor_to_pil(synthetic_image).save(image_save_path, quality=95)

                        # Annotations
                        anns_save_path = os.path.join(save_dir, "annotations", f"image_{self.rank}_{batch_counter}_{image_counter}_{k}.pkl")
                        with open(anns_save_path, 'wb') as f:
                            pickle.dump(anns_joined[k], f, protocol=pickle.HIGHEST_PROTOCOL)
                            
                    batch_counter += 1
                
            image_counter += 1
    
    def populate_parking_lot(self, annotations, set_type, scaling_factor):
        assert 'slots' in annotations.keys(), "Slots key missing in the parking lot annotations dictionary!"
        
        # Lists holding meshes and offsets
        parked_meshes = []
        parked_offsets = []
        
        # Extract image height and width
        height = self.image_size
        width = self.image_size
        
        # Extract the total number of slots
        n_slots = annotations['slots'].shape[0]
        parking_scenarios = self.compute_all_parking_scenarios(n_slots)
        for parking_scenario in parking_scenarios:
            parking_locations = annotations['slots'][[parking_scenario]][0]
            
            # Sample the required number of meshes
            n_meshes = parking_locations.shape[0]
            meshes = random.choices(self.meshes[set_type], k=n_meshes)
            
            # Convert locations to offsets
            parking_offsets = np.empty(shape=(0, 2))
            parking_angles = np.array([parking_location[2] for parking_location in parking_locations])
            for parking_location in parking_locations:
                # NOTE: parking locations are saved in (x, y, angle) format!
                parking_offsets = np.concatenate((
                    parking_offsets, 
                    np.expand_dims(np.array([-1 + 2 * parking_location[0] / width, -1 + 2 * parking_location[1] / height]), 0)
                ))
            
            # Move and rotate the meshes to the specified locations in the image
            meshes = self.move_and_rotate_meshes(meshes, parking_offsets, parking_angles, scaling_factor)
            
            # Join meshes as a scene
            parked_meshes.append(join_meshes_as_scene(meshes))
            parked_offsets.append(parking_offsets)
        
        return parked_meshes, parked_offsets
    
    def compute_all_parking_scenarios(self, n_slots):
        """
        Given the total number of parking slots available, this function returns indices of occupied parking slots for every single possible scenario of occupancy of
        the parking lot.
        
        inputs:
            - n_slots (int): number of available parking slots
        outputs:
            - parking_scenarios (List[int]): list of active indices, where each element of the list describes one unique scenario
        """
        scenarios_list = []
        indices = list(range(n_slots))
        for L in range(len(indices) + 1):
            for scenario in itertools.combinations(indices, L):
                if len(scenario) > 0:
                    scenarios_list.append(scenario)
        
        return scenarios_list
    
    def move_and_rotate_meshes(self, meshes, offsets, angles, scaling_factor):
        assert len(meshes) == offsets.shape[0] == angles.shape[0]
        for mesh, offset, angle in zip(meshes, offsets, angles):
            # Apply rotation
            mesh_rotation = euler_angles_to_matrix(torch.tensor([0, angle, 0]), convention="XYZ")
            mesh_rotation = torch.matmul(mesh_rotation.float(), mesh.verts_packed().data.T).T - mesh.verts_packed()
            mesh.offset_verts_(vert_offsets_packed=mesh_rotation)

            # Apply random translation (forcing the center of the vehicle to stay in the image)
            mesh_dx = -offset[0] / scaling_factor.item()
            mesh_dz = -offset[1] / scaling_factor.item()

            # Center the mesh before applying translation
            mesh_dx -= torch.mean(mesh.verts_padded(), dim=1)[0][0].item()
            mesh_dz -= torch.mean(mesh.verts_padded(), dim=1)[0][2].item()

            # Apply the translation
            mesh_translation = torch.tensor([mesh_dx, 0, mesh_dz]) * torch.ones(size=mesh.verts_padded().shape[1:])
            mesh_translation = mesh_translation.float()
            mesh.offset_verts_(vert_offsets_packed=mesh_translation)
        
        return meshes
    
    def prepare_dataloaders(self, pin_memory=False, num_workers=0):
        # Initialize the dataset
        train_set = ParkingDataset(os.path.join(self.cfg['DATA_DIR'], "train"), device=self.device)
        val_set = ParkingDataset(os.path.join(self.cfg['DATA_DIR'], "validation"), device=self.device)
        test_set = ParkingDataset(os.path.join(self.cfg['DATA_DIR'], "test"), device=self.device)
        
        # Retain required number of images
        if self.cfg['NUM_IMAGES']['ENABLE']:
            # Retain only a subset of the original datasets
            train_set.retain_n_images(self.cfg['NUM_IMAGES']['TRAIN'])
            val_set.retain_n_images(self.cfg['NUM_IMAGES']['VALIDATION'])
            test_set.retain_n_images(self.cfg['NUM_IMAGES']['TEST'])
        
        if self.cfg['NUM_GPUS'] == 1:
            # Initialize the dataloaders
            logger.info("Initializing the train loader")
            train_loader = DataLoader(train_set, batch_size=self.cfg['BATCH_SIZE'], collate_fn=collate_fn)
            logger.info("Initializing the validation loader")
            val_loader = DataLoader(val_set, batch_size=self.cfg['BATCH_SIZE'], collate_fn=collate_fn)
            logger.info("Initializing the test loader")
            test_loader = DataLoader(test_set, batch_size=self.cfg['BATCH_SIZE'], collate_fn=collate_fn)
        else:
            # Initialize the samplers
            train_sampler = DistributedSampler(train_set, num_replicas=self.world_size, rank=self.rank, shuffle=self.cfg['SHUFFLE'], drop_last=False)
            val_sampler = DistributedSampler(val_set, num_replicas=self.world_size, rank=self.rank, shuffle=self.cfg['SHUFFLE'], drop_last=False)
            test_sampler = DistributedSampler(test_set, num_replicas=self.world_size, rank=self.rank, shuffle=self.cfg['SHUFFLE'], drop_last=False)

            # Initialize the dataloaders
            logger.info("Initializing the train loader")
            train_loader = DataLoader(train_set, batch_size=self.cfg['BATCH_SIZE'], pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=train_sampler, collate_fn=collate_fn)
            logger.info("Initializing the validation loader")
            val_loader = DataLoader(val_set, batch_size=self.cfg['BATCH_SIZE'], pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=val_sampler, collate_fn=collate_fn)
            logger.info("Initializing the test loader")
            test_loader = DataLoader(test_set, batch_size=self.cfg['BATCH_SIZE'], pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=test_sampler, collate_fn=collate_fn)
        
        return (train_loader, val_loader, test_loader)