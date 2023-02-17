import os
import json
import numpy as np
import copy
import cv2
import torch
import torch.nn as nn
import pickle
from random import randrange
import argparse
from torchvision.ops import boxes as box_ops
from torch import Tensor

from detectron2.structures import BoxMode, Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils

import time
import pdb

def register_LINZ(data_path, mode, debug_on=False):
    data_path = copy.deepcopy(data_path)
    data_path = os.path.join(data_path, mode)
    annotations_dir = os.path.join(data_path, "annotations")    # full path to the annotations directory
    images_dir = os.path.join(data_path, "images")              # full path to the images directory
    
    annotations_list = os.listdir(annotations_dir)              # list of image filenames
    images_list = os.listdir(images_dir)                        # list of annotations filenames
    
    # Initialize the return list where the dicts will be stored
    annotations_list = ['0003_0002_0003398.pkl'] if debug_on else annotations_list
    dataset_dicts = [None for _ in range(len(annotations_list))]
    
    # Loop through the images
    for idx, annotation_file in enumerate(annotations_list):
        record = {}
        
        # Record preliminary information about the image
        file_name = annotation_file.split('.')[0] + '.jpg'
        image_id = idx
        height, width = 384, 384
        
        record["file_name"] = os.path.join(images_dir, file_name)
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width
        
        # Record detections
        vehicles = []
        with open(os.path.join(annotations_dir, annotation_file), 'rb') as f:
            annotations = pickle.load(f)
        del annotations['unknown'] # delete the unknown vehicles from the dataset
        for vehicle_type in annotations.keys():
            for vehicle_coordinate in annotations[vehicle_type]:
                # Generate a bounding box of size 12x12 in the (x_min, y_min, x_max, y_max) format.
                # NOTE: annotations are stored in the (x, y) format!
                bbox = np.concatenate((vehicle_coordinate - 6, vehicle_coordinate + 6), axis=0)
                vehicle = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0
                }
                vehicles.append(vehicle) # appends a numpy array consisting of 2 values in (x, y) format
        record["annotations"] = vehicles
        
        # dataset_dicts.append(record)
        dataset_dicts[idx] = record
    
    return dataset_dicts

def setup_dataset(data_path, debug_on=False):
    for mode in ["train", "validation", "test"]:
        # Register the dataset
        DatasetCatalog.register("LINZ_" + mode, lambda mode_=mode : register_LINZ(data_path, mode_, debug_on))
        
        # Update the metadata
        MetadataCatalog.get("LINZ_" + mode).set(thing_classes=["vehicle"], evaluator_type="coco_point")