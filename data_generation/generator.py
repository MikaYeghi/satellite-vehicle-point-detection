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
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import euler_angles_to_matrix

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
        # mesh_rotation = euler_angles_to_matrix(torch.tensor([0, 0, 0]), convention="XYZ").to(self.device)
        mesh_rotation = torch.matmul(mesh_rotation, mesh.verts_packed().data.T).T - mesh.verts_packed()
        mesh.offset_verts_(vert_offsets_packed=mesh_rotation)
        
        # Apply random translation (forcing the center of the vehicle to stay in the image)
        mesh_dx = random.uniform(-1, 1)
        mesh_dz = random.uniform(-1, 1)
        
        # Compute the offset
        offset = np.array([-mesh_dx, -mesh_dz]) # To be in the (x, y) format on the image
        
        mesh_dx /= scaling_factor
        mesh_dz /= scaling_factor
        mesh_dx -= torch.mean(mesh.verts_padded(), dim=1)[0][0].item()
        mesh_dz -= torch.mean(mesh.verts_padded(), dim=1)[0][2].item()
        mesh_translation = torch.tensor([mesh_dx, 0, mesh_dz], device=self.device) * torch.ones(size=mesh.verts_padded().shape[1:], device=self.device)
        mesh.offset_verts_(vert_offsets_packed=mesh_translation)
        
        return (mesh.clone(), offset)
    
    def randomly_place_meshes(self, meshes, scaling_factors):
        offsets = [None for _ in range(len(meshes))]
        for i in range(len(meshes)):
            meshes[i], offsets[i] = self.randomly_move_and_rotate_mesh(meshes[i], scaling_factors[i])
        return meshes, offsets
    
    def construct_annotations_files(self, locations):
        annotations = []
        
        for locations_ in locations:
            annotations_ = {
                "van_rv": np.empty(shape=(0, 2)),
                "truck": np.empty(shape=(0, 2)),
                "bus": np.empty(shape=(0, 2)),
                "trailer_small": np.empty(shape=(0, 2)),
                "specialized": np.empty(shape=(0, 2)),
                "trailer_large": np.empty(shape=(0, 2)),
                "unknown": np.empty(shape=(0, 2)),
                "small": locations_.unsqueeze(0).cpu().numpy()
            }
            annotations.append(annotations_)
    
        return annotations
    
    def generate(self):
        # Check that the save directories exist
        Path(os.path.join(self.cfg['SAVE_DIR'], "train", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "train", "annotations")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "validation", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "validation", "annotations")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "test", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "test", "annotations")).mkdir(parents=True, exist_ok=True)
        
        # Initialize the dataloader
        logger.info("Initializing the train loader")
        train_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "train"))
        train_loader = DataLoader(train_set, batch_size=self.cfg['BATCH_SIZE'])
        logger.info("Initializing the val loader")
        val_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "validation"))
        val_loader = DataLoader(val_set, batch_size=self.cfg['BATCH_SIZE'])
        logger.info("Initializing the test loader")
        test_set = GeneratorDataset(os.path.join(self.cfg['DATA_DIR'], "test"))
        test_loader = DataLoader(test_set, batch_size=self.cfg['BATCH_SIZE'])
        
        # Generate train/val/test sets
        self.generate_set("train", train_loader)
        self.generate_set("validation", val_loader)
        self.generate_set("test", test_loader)
        
        logger.info("Dataset generation finished!")
        
    def generate_set(self, set_type, dataloader):
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
            lights_directions = torch.rand(batch_size, 3) * 2 - 1
            lights_directions[:, 1] = -1
            scaling_factors = torch.rand(batch_size, 1) * (0.06 - 0.04) + 0.04
            intensities = torch.rand(batch_size, 1) * (2 - 0.5) + 0.5
            
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