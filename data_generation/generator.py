import os
import glob
import math
import torch
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from random import shuffle
from functools import reduce
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.transforms import euler_angles_to_matrix
from torch.utils.data.distributed import DistributedSampler
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, MeshRenderer, SoftSilhouetteShader, FoVOrthographicCameras, look_at_view_transform

from rendering.renderer import Renderer
from .dataset import GeneratorDataset
from .utils import sample_random_elev_azimuth

from logger import get_logger
logger = get_logger("Data generator logger")

import pdb

class Generator:
    def __init__(self, cfg, rank, world_size, image_size=384, device='cpu'):
        self.cfg = cfg
        self.image_size = image_size
        self.tensor_to_pil = transforms.ToPILImage()
        self.renderer = Renderer(device)
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # Load the meshes, already split into train/val/test meshes
        self.meshes = self.load_trainvaltest_meshes(cfg['MESHES_DIR'], cfg['MESHES_TRAINVALTEST_SPLIT'])
    
        # Load the distribution of vehicles if needed
        self.load_num_vehicles_dist()
    
    def load_trainvaltest_meshes(self, meshes_dir, trainvaltest_split, device='cpu'):
        logger.info(f"Loading meshes from {meshes_dir}")
        
        # Load the meshes
        meshes = []
        obj_paths = glob.glob(meshes_dir + "/*.obj")
        for obj_path in tqdm(obj_paths):
            mesh = load_objs_as_meshes([obj_path], device=device)[0]
            meshes.append(mesh)
        
        # Split into train/validation/test meshes
        assert len(trainvaltest_split) == 3, "Train/validation/test split of the meshes must contain 3 numbers!"
        assert sum(trainvaltest_split) == 1, "Train/validation/test split of the meshes must sum up to 1!"
        shuffle(meshes)
        n_meshes = len(meshes)
        train_meshes_idx = int(trainvaltest_split[0] * n_meshes)
        val_meshes_idx = int((trainvaltest_split[0] + trainvaltest_split[1]) * n_meshes)
        train_meshes = meshes[:train_meshes_idx]
        val_meshes = meshes[train_meshes_idx:val_meshes_idx]
        test_meshes = meshes[val_meshes_idx:]
        
        # Record the meshes
        meshes = {
            "train": train_meshes,
            "validation": val_meshes,
            "test": test_meshes
        }
        
        return meshes
    
    def load_num_vehicles_dist(self):
        if not self.cfg['MULTIVEHICLE']['ENABLE']:
            distribution = None
        else:
            if self.cfg['MULTIVEHICLE']['LOCATION_SAMPLING'] == 'random':
                distribution = None
            elif self.cfg['MULTIVEHICLE']['LOCATION_SAMPLING'] == 'real':
                """
                Load a pickle file that contains a dictionary in the following format:
                {num_vehicles: occurence_frequency}_i for i = 1, ..., N
                This dictionary describes an unnormalized distribution, that needs to be normalized.
                """
                with open(self.cfg['MULTIVEHICLE']['LOCATION_SAMPLING_FILE'], 'rb') as f:
                    distribution = pickle.load(f)
                
                # Find the maximum value
                max_freq = max(distribution.values())

                # Divide each value by the maximum value to normalize
                distribution = {k: v / max_freq for k, v in distribution.items()}
            else:
                logger.critical("LOCATION_SAMPLING parameter must be one of the following: random, real.")
                raise NotImplementedError
        self.num_vehicles_dist = distribution
    
    def place_matrix(self, meshes, scaling_factor, distance, elevation, azimuth, intensity, matrix_info):
        # Place each mesh in its cell
        matrix_height = matrix_info['height']
        matrix_width = matrix_info['width']
        step_dw = self.cfg['MULTIVEHICLE']['MATRIX']['WIDTH_SHIFT']
        step_dh = self.cfg['MULTIVEHICLE']['MATRIX']['HEIGHT_SHIFT']
        
        # Center the matrix before applying translation to it
        center_shift_dx = (matrix_width - 1) * step_dw / 2
        center_shift_dz = (matrix_height - 1) * step_dh / 2
        
        # Random translation
        matrix_shift_dx = random.uniform(-1, 1)
        matrix_shift_dz = random.uniform(-1, 1)
        
        # Random rotation (applied to the entire matrix)
        matrix_rotation = random.uniform(0, 2 * math.pi)
        rotation_cos = math.cos(matrix_rotation)
        rotation_sin = math.sin(matrix_rotation)
        
        # Random rotation applied to each vehicle (uniform across all vehicles)
        meshes_rotation = euler_angles_to_matrix(torch.tensor([0, random.uniform(0, -matrix_rotation + 2 * math.pi), 0]), convention="XYZ").to(self.device)
        
        offsets = []
        assert len(meshes) == matrix_height * matrix_width
        matrix_meshes = []
        
        k = 0
        for i in range(matrix_height):
            for j in range(matrix_width):
                mesh = meshes[k]
                
                # Get dx and dz
                mesh_dx = j * step_dw - center_shift_dx
                mesh_dz = i * step_dh - center_shift_dz
                
                # Get the original location of the mesh
                mesh_dx_original = 0 + mesh_dx
                mesh_dz_original = 0 + mesh_dz
                
                # Apply rotation to the matrix
                mesh_dx = mesh_dx_original * rotation_cos - mesh_dz_original * rotation_sin
                mesh_dz = mesh_dz_original * rotation_cos + mesh_dx_original * rotation_sin
                
                # Apply translation to the matrix
                mesh_dx += matrix_shift_dx
                mesh_dz += matrix_shift_dz
                
                # Compute the offset
                offset = np.array([-mesh_dx, -mesh_dz]) # To be in the (x, y) format on the image

                mesh_dx /= scaling_factor
                mesh_dz /= scaling_factor

                # Apply mesh rotation
                mesh_rotation = torch.matmul(meshes_rotation, mesh.verts_packed().data.T).T - mesh.verts_packed()
                mesh.offset_verts_(vert_offsets_packed=mesh_rotation)
                
                # Center the mesh before applying translation
                mesh_dx -= torch.mean(mesh.verts_padded(), dim=1)[0][0].item()
                mesh_dz -= torch.mean(mesh.verts_padded(), dim=1)[0][2].item()

                # Apply the translation
                mesh_translation = torch.tensor([mesh_dx, 0, mesh_dz], device=self.device) * torch.ones(size=mesh.verts_padded().shape[1:], device=self.device)
                mesh.offset_verts_(vert_offsets_packed=mesh_translation)
                offsets.append(offset)
                matrix_meshes.append(mesh.clone())
                k += 1
        
        return (matrix_meshes, offsets)
    
    def randomly_move_and_rotate_mesh(self, mesh, scaling_factor):
        # Apply random rotation
        mesh_rotation = euler_angles_to_matrix(torch.tensor([0, random.uniform(0, 2 * math.pi), 0]), convention="XYZ").to(self.device)
        mesh_rotation = torch.matmul(mesh_rotation, mesh.verts_packed().data.T).T - mesh.verts_packed()
        mesh.offset_verts_(vert_offsets_packed=mesh_rotation)
        
        # Apply random translation (forcing the center of the vehicle to stay in the image)
        mesh_dx = random.uniform(-1, 1)
        mesh_dz = random.uniform(-1, 1)
        
        # Compute the offset
        offset = np.array([-mesh_dx, -mesh_dz]) # To be in the (x, y) format on the image
        
        mesh_dx /= scaling_factor
        mesh_dz /= scaling_factor
        
        # Center the mesh before applying translation
        mesh_dx -= torch.mean(mesh.verts_padded(), dim=1)[0][0].item()
        mesh_dz -= torch.mean(mesh.verts_padded(), dim=1)[0][2].item()
        
        # Apply the translation
        mesh_translation = torch.tensor([mesh_dx, 0, mesh_dz], device=self.device) * torch.ones(size=mesh.verts_padded().shape[1:], device=self.device)
        mesh.offset_verts_(vert_offsets_packed=mesh_translation)
        
        return (mesh.clone(), offset)
    
    def randomly_move_and_rotate_meshes(self, meshes, scaling_factor, distance, elevation, azimuth, intensity):
        invalid_image = True
        
        # Create the silhouette renderer
        ambient_color = ((0.05, 0.05, 0.05),)
        diffuse_color = intensity * torch.tensor([1.0, 1.0, 1.0], device=self.device).unsqueeze(0)
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        cameras = FoVOrthographicCameras(
            device=self.device, 
            R=R,
            T=T, 
            scale_xyz=((scaling_factor, scaling_factor, scaling_factor),)
        ) 
        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size=self.cfg['IMAGE_SIZE'], 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
        )
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )
        
        while invalid_image:
            offsets = []
            silhouettes = []
            
            for i in range(len(meshes)):
                meshes[i], offset = self.randomly_move_and_rotate_mesh(meshes[i], scaling_factor)
                silhouette = silhouette_renderer(meshes[i], cameras=cameras)
                silhouette = (silhouette[..., 3] > 0.5).float()
                silhouettes.append(silhouette)
                offsets.append(offset)
            
            # Check whether any of the meshes intersect
            if torch.any(reduce(lambda x, y: x + y, silhouettes) > 1.0):
                invalid_image = True
            else:
                invalid_image = False
        
        return (meshes, offsets)
    
    def randomly_place_meshes(self, meshes, scaling_factors):
        offsets = [None for _ in range(len(meshes))]
        for i in range(len(meshes)):
            meshes[i], offsets[i] = self.randomly_move_and_rotate_mesh(meshes[i], scaling_factors[i])
        return meshes, offsets
    
    def randomly_place_meshes_multi(self, meshes_list, scaling_factors, distances, elevations, azimuths, intensities, matrix_info):
        assert len(meshes_list) == len(scaling_factors)
        meshes = []
        offsets = []
        
        i = 0
        for i in range(len(meshes_list)):
            meshes_ = meshes_list[i]
            scaling_factor = scaling_factors[i]
            distance = distances[i]
            elevation = elevations[i]
            azimuth = azimuths[i]
            intensity = intensities[i]
            matrix_info_ = matrix_info[i]
            
            if matrix_info_['is_matrix']:
                meshes_, offsets_ = self.place_matrix(meshes_, scaling_factor, distance, elevation, azimuth, intensity, matrix_info[i])
            else:
                meshes_, offsets_ = self.randomly_move_and_rotate_meshes(meshes_, scaling_factor, distance, elevation, azimuth, intensity)
            
            meshes.append(meshes_)
            offsets.append(offsets_)
        return meshes, offsets
    
    def construct_annotations_files(self, locations_batch):
        annotations_batch = []
        
        for locations in locations_batch:
            annotations = {
                "van_rv": np.empty(shape=(0, 2)),
                "truck": np.empty(shape=(0, 2)),
                "bus": np.empty(shape=(0, 2)),
                "trailer_small": np.empty(shape=(0, 2)),
                "specialized": np.empty(shape=(0, 2)),
                "trailer_large": np.empty(shape=(0, 2)),
                "unknown": np.empty(shape=(0, 2)),
                "small": locations.cpu().numpy()
            }
            annotations_batch.append(annotations)
    
        return annotations_batch
    
    def generate(self):
        if self.cfg['MULTIVEHICLE']['ENABLE']:
            logger.info("Generating a dataset: multi-vehicle setup")
            self.generate_multi()
        else:
            logger.info("Generating a dataset: single vehicle setup")
            self.generate_single()
    
    def generate_single(self):
        """
        This function generates a dataset of synthetic images, where each image contains 
        only one vehicle.
        """
        # Check that the save directories exist
        Path(os.path.join(self.cfg['SAVE_DIR'], "train", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "train", "annotations")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "validation", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "validation", "annotations")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "test", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "test", "annotations")).mkdir(parents=True, exist_ok=True)
        
        # Get the dataloaders
        train_loader, val_loader, test_loader = self.prepare_dataloaders()
        
        # Generate train/val/test sets
        self.generate_set_single("train", train_loader)
        self.generate_set_single("validation", val_loader)
        self.generate_set_single("test", test_loader)
        
        logger.info("Dataset generation finished!")
        
    def generate_set_single(self, set_type, dataloader):
        assert set_type in ['train', 'validation', 'test'], "Set type must be one of train/val/test!"
        data_dir = os.path.join(self.cfg['DATA_DIR'], set_type)
        save_dir = os.path.join(self.cfg['SAVE_DIR'], set_type)
        image_counter = 0
        logger.info(f"Generating {set_type} dataset")
        
        # Populate images with rendered vehicles
        for images_batch in tqdm(dataloader):
            batch_size = len(images_batch)
            
            # Randomly select meshes
            meshes = random.choices(self.meshes[set_type], k=batch_size)
            
            # Move image and mesh to the device
            meshes = [mesh.to(self.device) for mesh in meshes]
            images_batch = images_batch.to(self.device)
            
            # Sample rendering parameters (TO DO: sample light direction)
            distances = [5.0 for _ in range(batch_size)]
            elevations = [90 for _ in range(batch_size)]
            azimuths = [0 for _ in range(batch_size)]
            lights_directions = torch.rand(batch_size, 3, device=self.device) * 2 - 1
            lights_directions[:, 1] = -1
            scaling_factors = torch.rand(batch_size, 1, device=self.device) * (0.06 - 0.04) + 0.04
            intensities = torch.rand(batch_size, 1, device=self.device) * (2 - 0.5) + 0.5
            
            # Randomly move and rotate the meshes
            meshes, offsets = self.randomly_place_meshes(
                meshes,
                scaling_factors
            )
            
            # Convert offset to coordinates. Note: required shape (B, 2).
            locations = torch.tensor(offsets, device=self.device)
            locations = (1 + locations) * self.image_size / 2
            
            # Render the images
            synthetic_images = self.renderer.render_batch(
                meshes, 
                images_batch, 
                elevations, 
                azimuths,
                lights_directions,
                scaling_factors=scaling_factors,
                intensities=intensities,
                distances=distances,
                image_size=self.image_size
            )
            
            # Construct the annotations
            annotations = self.construct_annotations_files(locations)
            
            # Save the images and annotations
            for k in range(len(synthetic_images)):
                # Image
                synthetic_image = synthetic_images[k]
                image_save_path = os.path.join(save_dir, "images", f"image_{image_counter}_{k}.jpg")
                self.tensor_to_pil(synthetic_image).save(image_save_path, quality=95)
            
                # Annotations
                anns_save_path = os.path.join(save_dir, "annotations", f"image_{image_counter}_{k}.pkl")
                with open(anns_save_path, 'wb') as f:
                    pickle.dump(annotations[k], f, protocol=pickle.HIGHEST_PROTOCOL)
                
            image_counter += 1
        
    def generate_multi(self):
        """
        This function generates a dataset of synthetic images, where each image contains 
        multiple vehicle.
        """
        # Check that the save directories exist
        Path(os.path.join(self.cfg['SAVE_DIR'], "train", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "train", "annotations")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "validation", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "validation", "annotations")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "test", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "test", "annotations")).mkdir(parents=True, exist_ok=True)
        
        # Get the dataloaders
        train_loader, val_loader, test_loader = self.prepare_dataloaders()

        # Generate train/val/test sets
        self.generate_set_multi("train", train_loader)
        self.generate_set_multi("validation", val_loader)
        self.generate_set_multi("test", test_loader)

        logger.info("Dataset generation finished!")
        
    def generate_set_multi(self, set_type, dataloader):
        assert set_type in ['train', 'validation', 'test'], "Set type must be one of train/val/test!"
        data_dir = os.path.join(self.cfg['DATA_DIR'], set_type)
        save_dir = os.path.join(self.cfg['SAVE_DIR'], set_type)
        image_counter = 0
        logger.info(f"Generating {set_type} dataset")
        
        # Populate images with rendered vehicles
        for images_batch in tqdm(dataloader):
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
        
    def sample_n_vehicles(self, batch_size):
        n_vehicles_max = self.cfg['MULTIVEHICLE']['N_VEHICLES_MAX']
        
        if self.cfg['MULTIVEHICLE']['LOCATION_SAMPLING'] == 'random':
            n_vehicles_list = [random.randint(1, n_vehicles_max) for _ in range(batch_size)]
        elif self.cfg['MULTIVEHICLE']['LOCATION_SAMPLING'] == 'real':
            n_vehicles_list = []
            for _ in range(batch_size):
                accepted = False
                while not accepted:
                    num_vehicles = random.choice(list(self.num_vehicles_dist.keys()))
                    accepted = (random.uniform(0, 1) < self.num_vehicles_dist[num_vehicles])
                n_vehicles_list.append(num_vehicles)
        else:
            raise NotImplementedError
        
        return n_vehicles_list
    
    def prepare_dataloaders(self, pin_memory=False, num_workers=0):
        # Initialize the dataset
        train_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "train"), device=self.device)
        val_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "validation"), device=self.device)
        test_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "test"), device=self.device)
        
        # Retain required number of images
        if self.cfg['NUM_IMAGES']['ENABLE']:
            # Retain only a subset of the original datasets
            train_set.retain_n_images(self.cfg['NUM_IMAGES']['TRAIN'])
            val_set.retain_n_images(self.cfg['NUM_IMAGES']['VALIDATION'])
            test_set.retain_n_images(self.cfg['NUM_IMAGES']['TEST'])
        
        # Initialize the samplers
        train_sampler = DistributedSampler(train_set, num_replicas=self.world_size, rank=self.rank, shuffle=self.cfg['SHUFFLE'], drop_last=False)
        val_sampler = DistributedSampler(val_set, num_replicas=self.world_size, rank=self.rank, shuffle=self.cfg['SHUFFLE'], drop_last=False)
        test_sampler = DistributedSampler(test_set, num_replicas=self.world_size, rank=self.rank, shuffle=self.cfg['SHUFFLE'], drop_last=False)
        
        # Initialize the dataloaders
        logger.info("Initializing the train loader")
        train_loader = DataLoader(train_set, batch_size=self.cfg['BATCH_SIZE'], pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=train_sampler)
        logger.info("Initializing the validation loader")
        val_loader = DataLoader(val_set, batch_size=self.cfg['BATCH_SIZE'], pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=val_sampler)
        logger.info("Initializing the test loader")
        test_loader = DataLoader(test_set, batch_size=self.cfg['BATCH_SIZE'], pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=test_sampler)
        
        return (train_loader, val_loader, test_loader)