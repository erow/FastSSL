"""example usage:
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/
write_dataset train 500 0.50 90
write_path=$WRITE_DIR/train500_0.5_90.ffcv
echo "Writing ImageNet train dataset to ${write_path}"
python examples/write_dataset.py \
    --cfg.data_dir=$IMAGENET_DIR \
    --cfg.write_path=$write_path \
    --cfg.max_resolution=500 \
    --cfg.write_mode=smart \
    --cfg.compress_probability=0.50 \
    --cfg.jpeg_quality=90
"""
import os
from PIL import Image
from torch.utils.data import Subset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
import torchvision
from torchvision.datasets import  ImageFolder
import torchvision.datasets as torch_datasets

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config
import cv2
import numpy as np

import torch
from torchvision import transforms
import timm
import random 


Section('cfg', 'arguments to give the writer').params(
    dataset=Param(And(str, OneOf(['cifar', 'imagenet'])), 'Which dataset to write', default='imagenet'),
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    write_path=Param(str, 'Where to write the new dataset', required=True),
    write_mode=Param(str, 'Mode: raw, smart or jpg', required=False, default='smart'),
    max_resolution=Param(int, 'Max image side length. 0 any size.', required=False,default=0),
    num_workers=Param(int, 'Number of workers to use', default=16),
    chunk_size=Param(int, 'Chunk size for writing', default=100),
    jpeg_quality=Param(float, 'Quality of jpeg images', default=90),
    subset=Param(float, 'How many images to use (the fraction of the dataset, 1 for all)', default=0.1),
    compress_probability=Param(float, 'compress probability', default=0.5),
    threshold=Param(int, 'threshold for smart mode to compress by jpeg', default=286432),
    proxy=Param(str, 'proxy model to use', default='resnet18'),
    sub_mode=Param(And(str, OneOf(['sas', 'random'])), 'Subset mode', default='sas'),
)

@section('cfg')
@param('dataset')
@param('data_dir')
@param('write_path')
@param('max_resolution')
@param('num_workers')
@param('chunk_size')
@param('subset')
@param('jpeg_quality')
@param('write_mode')
@param('compress_probability')
@param('threshold')
@param('proxy')
@param('sub_mode')
def main(dataset, data_dir, write_path, max_resolution, num_workers,
         chunk_size, subset, jpeg_quality, write_mode,
         compress_probability, threshold, proxy,sub_mode):
    
    if dataset == 'imagenet':
        my_dataset = ImageFolder(root=data_dir)
    elif dataset == 'cifar':        
        my_dataset = torch_datasets.CIFAR10(root=data_dir, train=True, download=True)
    else:
        raise ValueError('Unknown dataset')
    
    if sub_mode=="sas":
        device = "cuda:0"

        # Approximate Latent Classes
        from sas.approx_latent_classes import clip_approx
        from sas.subset_dataset import SASSubsetDataset
        ds_size = len(my_dataset)
        rand_labeled_examples_indices = random.sample(range(ds_size), 10000)
        rand_labeled_examples_labels = [my_dataset.samples[i][1] for i in rand_labeled_examples_indices]

        if os.path.exists('/tmp/clip_partition.npy'):
            partition = np.load('/tmp/clip_partition.npy', allow_pickle=True).item()
        else:
            partition = clip_approx(
                img_trainset=my_dataset,
                labeled_example_indices=rand_labeled_examples_indices, 
                labeled_examples_labels=rand_labeled_examples_labels,
                num_classes=1000,
                device=device, verbose=True
            )
            np.save('/tmp/clip_partition.npy', partition)

        # Get Subset
        proxy_model = timm.create_model(proxy, pretrained=True).to(device)
        
        augmentation_distance = None
        if os.path.exists('/tmp/augmentation_distance.npy'):
            augmentation_distance = np.load('/tmp/augmentation_distance.npy',allow_pickle=True).item()

        default_tfms = my_dataset.transform
        my_dataset.transform = torchvision.transforms.Compose([
            transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        subset_dataset = SASSubsetDataset(
            dataset=my_dataset,
            subset_fraction=subset,
            num_downstream_classes=1000,
            device=device,
            proxy_model=proxy_model,
            approx_latent_class_partition=partition,
            augmentation_distance = augmentation_distance,
            verbose=False
        )
        if augmentation_distance is None:
            np.save('/tmp/augmentation_distance.npy', subset_dataset.augmentation_distance)
        my_dataset.transform = default_tfms
        subset_indecies = subset_dataset.subset_indices
    elif sub_mode=="random":
        subset_indecies = random.sample(range(len(my_dataset)), int(subset*len(my_dataset)))
    else:
        raise ValueError('Unknown sub_mode')
    my_dataset = Subset(my_dataset, subset_indecies)

    writer = DatasetWriter(write_path, {
        'image': RGBImageField(write_mode=write_mode,
                               max_resolution=None if max_resolution==0 else max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality,
                               smart_threshold=threshold),
        'label': IntField(),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(my_dataset, chunksize=chunk_size,shuffle_indices=False)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    
    args=config.get().cfg
    assert args.write_path.endswith('.ffcv'), 'write_path must end with .ffcv'
    file=open(args.write_path.replace(".ffcv",".meta"), 'w')
    file.write(str(args.__dict__))
    main()
