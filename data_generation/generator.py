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
from torchvision import transforms
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import euler_angles_to_matrix

from rendering.renderer import Renderer
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
    
    def randomly_place_meshes(self, meshes, distance, elevation, azimuth, lights_direction, scaling_factor, intensity):
        return self.randomly_move_and_rotate_mesh(meshes[0], scaling_factor)
    
    def construct_annotations_file(self, locations):
        annotations = {
            "van_rv": np.empty(shape=(0, 2)),
            "truck": np.empty(shape=(0, 2)),
            "bus": np.empty(shape=(0, 2)),
            "trailer_small": np.empty(shape=(0, 2)),
            "specialized": np.empty(shape=(0, 2)),
            "trailer_large": np.empty(shape=(0, 2)),
            "unknown": np.empty(shape=(0, 2)),
            "small": locations
        }
    
        return annotations
    
    def generate(self):
        # Check that the save directories exist
        Path(os.path.join(self.cfg['SAVE_DIR'], "train", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "train", "annotations")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "validation", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "validation", "annotations")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "test", "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.cfg['SAVE_DIR'], "test", "annotations")).mkdir(parents=True, exist_ok=True)
        
        # Generate train/val/test sets
        self.generate_set("train")
        self.generate_set("validation")
        self.generate_set("test")
        
        logger.info("Dataset generation finished!")
        
    def generate_set(self, set_type):
        assert set_type in ['train', 'validation', 'test'], "Set type must be one of train/val/test!"
        data_dir = os.path.join(self.cfg['DATA_DIR'], set_type)
        save_dir = os.path.join(self.cfg['SAVE_DIR'], set_type)
        image_counter = 0
        logger.info(f"Generating {set_type} dataset")
        
        # Load annotation files
        pil_to_tensor = transforms.ToTensor()
        annotation_files = glob.glob(os.path.join(data_dir, "annotations") + "/*.pkl")
        
        # Populate images with rendered vehicles
        for annotations_file in tqdm(annotation_files):
            with open(annotations_file, 'rb') as f:
                annotations = pickle.load(f)
            del annotations['unknown'] # delete the unknown vehicles from the dataset
            
            # Check if this annotation file is empty. If not, skip it.
            not_empty = any(v.shape[0] > 0 for v in annotations.values())
            if not_empty:
                continue
                
            # Otherwise, load the image and add vehicles to it
            image_path = os.path.join(
                data_dir, 
                "images",
                annotations_file.split('/')[-1][:-4] + ".jpg"
            )
            image = pil_to_tensor(Image.open(image_path).convert('RGB'))
            
            # Randomly select a mesh
            mesh = random.choice(self.meshes[set_type])
            
            # Move image and mesh to the device
            mesh = mesh.to(self.device)
            image = image.to(self.device)
            
            # Sample rendering parameters (TO DO: sample light direction)
            distance = 5.0
            elevation, azimuth = 90, 0
            lights_direction = torch.tensor([random.uniform(-1, 1),-1.0,random.uniform(-1, 1)], device=self.device, requires_grad=True).unsqueeze(0)
            scaling_factor = random.uniform(0.04, 0.06)
            intensity = random.uniform(0.5, 2.0)
            
            # Randomly move and rotate the meshes
            mesh, offset = self.randomly_place_meshes(
                mesh, 
                distance, 
                elevation, 
                azimuth, 
                lights_direction, 
                scaling_factor, 
                intensity
            )
            
            # Convert offset to coordinates. Note: required shape (B, 2).
            locations = np.expand_dims(self.image_size / 2 * (1 + offset), 0)
            
            # Render the image
            synthetic_image = self.renderer.render(
                mesh, 
                image, 
                elevation, 
                azimuth,
                lights_direction,
                scaling_factor=scaling_factor,
                intensity=intensity,
                ambient_color=((0.05, 0.05, 0.05),),
                distance=distance,
                image_size=self.image_size
            )
            
            # Construct the annotations
            annotations = self.construct_annotations_file(locations)
            
            # Save the image
            image_save_path = os.path.join(save_dir, "images", f"image_{image_counter}.jpg")
            self.tensor_to_pil(synthetic_image.permute(2, 0, 1)).save(image_save_path, quality=95)
            
            # Save annotations
            anns_save_path = os.path.join(save_dir, "annotations", f"image_{image_counter}.pkl")
            with open(anns_save_path, 'wb') as f:
                pickle.dump(annotations, f, protocol=pickle.HIGHEST_PROTOCOL)
        
            image_counter += 1