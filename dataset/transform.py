"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms.v2 import Normalize, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, RandomApply, RandomGrayscale, Compose, ToDtype
import gin
from PIL import Image

from typing import List

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEFAULT_CROP_RATIO = 224/256

class ToDevice(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x:torch.Tensor):
        return x.to(self.device,non_blocking=True)
    
class GaussianBlur(nn.Module):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(nn.Module):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

@gin.configurable()
class SimpleAugmentation(nn.Module):
    def __init__(self,img_size=224,scale=(0.2, 1.0),
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
         # simple augmentation
        self.transforms = Compose([
                RandomResizedCrop(img_size, scale=scale, interpolation=Image.BICUBIC),  # 3 is bicubic
                RandomHorizontalFlip(),
                ToTensor(),
                # ToDevice('cuda'),
                Normalize(mean=mean,std=std)])
    def forward(self,x):
        return self.transforms(x)
    
    def change_resolution(self,img_size):
        decoder = self.transforms[0]
        decoder.size=(img_size,img_size)


@gin.configurable()
class DataAugmentationDINO(nn.Module):
    def __init__(self,img_size=224, global_crops_scale=(0.4, 1.), local_crops_scale=(0.05, 0.4), local_crops_number=8, color_jitter=True,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Multi-view data augmentation.

        Args:
            global_crops_scale (tuple, optional): _description_. Defaults to (0.4, 1.).
            local_crops_scale (tuple, optional): _description_. Defaults to (0.05, 0.4).
            local_crops_number (int, optional): _description_. Defaults to 8.
        
        Return:
            [2 x global views, local_crops_number x local views]
        """
        super().__init__()
        flip_and_color_jitter = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomApply(
                [ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            RandomGrayscale(p=0.2),
        ])

        normalize = Compose([
            ToTensor(),
            # ToDevice('cuda'),
            Normalize(mean, std),
        ])

        # first global crop
        self.global_transfo1 = Compose([
            RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(5),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = Compose([
            RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(5),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = Compose([
            RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            RandomApply([GaussianBlur(5)],p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops