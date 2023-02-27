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
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, MeshRenderer, SoftSilhouetteShader, FoVOrthographicCameras, look_at_view_transform

from rendering.renderer import Renderer
from .dataset import GeneratorDataset
from .utils import sample_random_elev_azimuth

from logger import get_logger
logger = get_logger("Data generator logger")

import pdb

class Generator:
    def __init__(self, cfg, image_size=384, device='cpu'):
        self.cfg = cfg
        self.image_size = image_size
        self.tensor_to_pil = transforms.ToPILImage()
        self.renderer = Renderer(device)
        self.device = device
        
        # Load the meshes, already split into train/val/test meshes
        self.meshes = self.load_trainvaltest_meshes(cfg['MESHES_DIR'], cfg['MESHES_TRAINVALTEST_SPLIT'])
    
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
    
    def randomly_place_meshes_multi(self, meshes_list, scaling_factors, distances, elevations, azimuths, intensities):
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
        
        # Initialize the dataloaders
        logger.info("Initializing the train loader")
        train_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "train"))
        train_loader = DataLoader(train_set, batch_size=self.cfg['BATCH_SIZE'])
        logger.info("Initializing the validation loader")
        val_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "validation"))
        val_loader = DataLoader(val_set, batch_size=self.cfg['BATCH_SIZE'])
        logger.info("Initializing the test loader")
        test_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "test"))
        test_loader = DataLoader(test_set, batch_size=self.cfg['BATCH_SIZE'])
        
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

        # Initialize the dataloaders
        logger.info("Initializing the train loader")
        train_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "train"))
        train_loader = DataLoader(train_set, batch_size=self.cfg['BATCH_SIZE'])
        logger.info("Initializing the validation loader")
        val_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "validation"))
        val_loader = DataLoader(val_set, batch_size=self.cfg['BATCH_SIZE'])
        logger.info("Initializing the test loader")
        test_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "test"))
        test_loader = DataLoader(test_set, batch_size=self.cfg['BATCH_SIZE'])

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
            
            # Move meshes to the device
            for i in range(len(n_vehicles_list)):
                for j in range(n_vehicles_list[i]):
                    meshes_batch_list[i][j] = meshes_batch_list[i][j].to(self.device)
            
            # Sample rendering parameters
            distances = [5.0 for _ in range(batch_size)]
            elevations = [90 for _ in range(batch_size)]
            azimuths = [0 for _ in range(batch_size)]
            lights_directions = torch.rand(batch_size, 3, device=self.device) * 2 - 1
            lights_directions[:, 1] = -1
            scaling_factors = torch.rand(batch_size, 1, device=self.device) * (0.06 - 0.04) + 0.04
            intensities = torch.rand(batch_size, 1, device=self.device) * (2 - 0.5) + 0.5
            
            # Randomly place the vehicles in each image
            meshes, offsets = self.randomly_place_meshes_multi(meshes_batch_list, scaling_factors, distances, elevations, azimuths, intensities)
            
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
                image_save_path = os.path.join(save_dir, "images", f"image_{image_counter}_{k}.jpg")
                self.tensor_to_pil(synthetic_image).save(image_save_path, quality=95)
            
                # Annotations
                anns_save_path = os.path.join(save_dir, "annotations", f"image_{image_counter}_{k}.pkl")
                with open(anns_save_path, 'wb') as f:
                    pickle.dump(annotations[k], f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                # DEBUG
                from matplotlib import pyplot as plt
                fig, ax = plt.subplots()
                ax.imshow(synthetic_image.permute(1, 2, 0).cpu().numpy())
                xs = annotations[k]['small'][:, 0]
                ys = annotations[k]['small'][:, 1]
                ax.plot(xs, ys, 'rx')
                fig.savefig(f"results/image_{image_counter}_{k}.png")
                plt.close(fig)
                
            image_counter += 1
        
    def sample_n_vehicles(self, batch_size):
        n_vehicles_max = self.cfg['MULTIVEHICLE']['N_VEHICLES_MAX']
        
        if self.cfg['MULTIVEHICLE']['LOCATION_SAMPLING'] == 'random':
            n_vehicles_list = [random.randint(1, n_vehicles_max) for _ in range(batch_size)]
        else:
            raise NotImplementedError
        
        return n_vehicles_list