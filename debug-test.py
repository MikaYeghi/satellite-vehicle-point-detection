import torch
import torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import numpy as np
import os, json, cv2, random, pdb
import matplotlib.pyplot as plt

cfg = get_cfg()
cfg.merge_from_file("output/config.yaml")
cfg.merge_from_list([
    "MODEL.WEIGHTS", "output/model_final.pth",
    "MODEL.RETINANET.SCORE_THRESH_TEST", "0.7"
])

predictor = DefaultPredictor(cfg)

im = cv2.imread("/var/storage/myeghiaz/Detection/SatDet-Real-384px-0.25m-debug/train/images/0002_0001_0007276.jpg")
print(im.shape)
plt.figure(figsize=(15,7.5))
plt.imshow(im[..., ::-1])

outputs = predictor(im[..., ::-1])

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.figure(figsize=(20,10))
plt.imshow(out.get_image()[..., ::-1][..., ::-1])
plt.savefig("sample.png")