import os
import glob
import random
import pickle
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor

from torch.utils.data import Dataset

from logger import get_logger
logger = get_logger("GeneratorDataset logger")

import pdb

class GeneratorDataset(Dataset):
    def __init__(self, data_dir, device='cpu'):
        super().__init__()
        
        self.data_dir = data_dir
        self.device = device
        
        self.transform = ToTensor()
        self.metadata = self.extract_metadata(data_dir)
        
        logger.info(f"Loaded a dataset with {self.__len__()} images.")
        
    def extract_metadata(self, data_dir):
        """
        This method extract metadata which described the detection dataset.
        
        Expecting the dataset should have the following structure:
        └── data_dir
            ├── annotations
            ├── images
        Each image in the "images" directory has an analogous annotation file in the "annotations" directory.
        For example: image_1234.jpg <-> image_1234.pkl.
        """
        metadata = []
        images_dir = os.path.join(data_dir, "images")
        annotations_dir = os.path.join(data_dir, "annotations")
        
        # Load the images and annotations list
        images_list = []
        formats_list = ['jpg', 'png']
        for image_format in formats_list:
            images_list += glob.glob(images_dir + f"/*.{image_format}")
        annotations_list = glob.glob(annotations_dir + "/*.pkl")
        if len(images_list) != len(annotations_list): logger.warning("Different number of images and annotation files!")
        
        # Load the metadata
        for image_path in tqdm(images_list):
            # Get the annotation file path
            file_code = image_path.split('/')[-1].split('.')[0]
            annotation_path = os.path.join(annotations_dir, f"{file_code}.pkl")
            
            with open(annotation_path, 'rb') as f:
                annotations = pickle.load(f)
            del annotations['unknown'] # delete the unknown vehicles from the dataset
            
            # If image is not empty -- skip it.
            if any([x.shape[0] > 0 for x in annotations.values()]):
                continue
            
            metadata_ = {
                "image_path": image_path,
                "annotations": annotations
            }
            
            metadata.append(metadata_)
        
        return metadata
    
    def retain_n_images(self, n):
        """
        This function randomly selects n images from the extracted dataset, and removes all other image metadata.
        """
        assert n > 0, "Number of retained images must be greater than 0!"
        original_length = len(self.metadata)
        if n < original_length:
            self.metadata = random.sample(self.metadata, n)
        logger.info(f"Removed {original_length - len(self.metadata)} images from the dataset. Using {len(self.metadata)} images.")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        data = self.metadata[idx]
        image_path = data['image_path']
        annotations = data['annotations']
        
        # Load the image
        image = Image.open(image_path)
        image = self.transform(image)
        
        # Move to the device
        image = image.to(self.device)
        
        return image
        