import os
import torch
import random
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch3d.structures import join_meshes_as_scene
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
            
            # Sample the number of vehicles for each image
            n_vehicles_list = self.sample_n_vehicles(batch_size)
            
            # Sample meshes for the batch
            meshes_batch_list = [random.choices(self.meshes[set_type], k=n_vehicles) for n_vehicles in n_vehicles_list]
            
            # Generate matrix meshes (parking lot analog)
            if self.cfg['MULTIVEHICLE']['MATRIX']['ENABLE']:
                matrix_prob = self.cfg['MULTIVEHICLE']['MATRIX']['PROBABILITY']
                matrix_info = [{'is_matrix': random.uniform(0, 1) < matrix_prob} for _ in range(batch_size)]
                for i in range(batch_size):
                    if matrix_info[i]['is_matrix']:
                        matrix_max_size = self.cfg['MULTIVEHICLE']['MATRIX']['MAX_SIZE']
                        matrix_height = random.randint(2, matrix_max_size)
                        matrix_width = random.randint(2, matrix_max_size)
                        meshes_batch_list[i] = random.choices(self.meshes[set_type], k=matrix_height*matrix_width)
                        n_vehicles_list[i] = matrix_height * matrix_width
                        matrix_info[i]['height'] = matrix_height
                        matrix_info[i]['width'] = matrix_width
            else:
                matrix_info = [{'is_matrix': False, 'height': 0, 'width': 0} for _ in range(batch_size)]
            
            # Move meshes to the device
            for i in range(len(n_vehicles_list)):
                for j in range(n_vehicles_list[i]):
                    meshes_batch_list[i][j] = meshes_batch_list[i][j].to(self.device)
            
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
            
            # Randomly place the vehicles in each image
            meshes, offsets = self.randomly_place_meshes_multi(meshes_batch_list, scaling_factors, distances, elevations, azimuths, intensities, matrix_info)
            
            # Convert offsets to locations
            locations_batch = []
            for offsets_ in offsets:
                locations = torch.tensor(offsets_, device=self.device)
                locations = (1 + locations) * self.image_size / 2
                locations_batch.append(locations)
            annotations = self.construct_annotations_files(locations_batch)
            
            # Join images in each scene into a single mesh
            meshes_joined = []
            for i in range(batch_size):
                meshes_joined.append(join_meshes_as_scene(meshes[i]).to(self.device))
            
            # Render the images
            synthetic_images = self.renderer.render_batch(
                meshes_joined, 
                images_batch, 
                elevations, 
                azimuths,
                lights_directions,
                scaling_factors=scaling_factors,
                intensities=intensities,
                distances=distances,
                image_size=self.image_size
            )
            
            # Save images and annotations
            for k in range(batch_size):
                # Image
                synthetic_image = synthetic_images[k]
                image_save_path = os.path.join(save_dir, "images", f"image_{self.rank}_{image_counter}_{k}.jpg")
                self.tensor_to_pil(synthetic_image).save(image_save_path, quality=95)
            
                # Annotations
                anns_save_path = os.path.join(save_dir, "annotations", f"image_{self.rank}_{image_counter}_{k}.pkl")
                with open(anns_save_path, 'wb') as f:
                    pickle.dump(annotations[k], f, protocol=pickle.HIGHEST_PROTOCOL)
                
            image_counter += 1
        
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