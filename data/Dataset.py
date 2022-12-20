# JW 
#   Dragonball
#       data
#           Dataset.py

#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from glob import glob
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import os
import glob
from PIL import Image


seed = 42


class CrackDataset(Dataset):
    def __init__(self, split = 'train'):
        super().__init__()
        path = os.path.join(os.path.dirname(__file__), 'crack_segmentation_dataset').replace('G1/dragon_ball', 'DragonBall')
        path = os.path.join(path, split, 'images')
        self.img_path = glob.glob(path + '/*.jpg')
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.image_size = (448, 448)
        
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        image_path = self.img_path[idx]
        label_path = image_path.replace('images', 'masks')
        
        image = Image.open(image_path)      
        mask = Image.open(label_path)

        image = self.transforms(image)
        mask = self.transforms(mask)
        mask[mask > 0] = 1

        if len(mask.unique()) > 2:
            error_msg = "error - mask is not binary"
            print(error_msg)
            return error_msg
        else:
            return image, mask

def split_Dataset(length, mode = 'mid'):
    ratio = 1
    if mode == 'mini':  ratio = 0.01
    elif mode == 'mid': ratio = 0.1
    train_size = int(ratio*length)
    return train_size, (length-train_size)

def get_loader(dataset, batch_size=16, num_workers=4, mode='max'):
    train_size, valid_size = split_Dataset(len(dataset), mode)
    dataset, drop_dataset = random_split(dataset, [train_size, valid_size])
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


"""
    #
    train_size = int(0.1*len(train_set))
    valid_size = len(train_set) - train_size
    train_set, valid_dataset = random_split(train_set, [train_size, valid_size])
    train_size = int(0.8*len(test_set))
    valid_size = len(test_set) - train_size
    test_set, valid_dataset = random_split(test_set, [train_size, valid_size])
"""