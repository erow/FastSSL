# A Guideline for Small Datasets

## Introduction
In this guideline, we will discuss the challenges of working with small datasets and provide some strategies to overcome these challenges. Small datasets provide affordable and quick access to validate your idea, but the results do bot necessarily generalize to larger datasets. Nevertheless, small datasets are useful for understanding the learning algorithms and debugging the code. Especially, when you are a beginner in machine learning without a lot GPU resources, small datasets are a good starting point. Our goal is to provide efficient training recipes for small datasets with competitive performance. Our discussion focuses on CIFAR10 and mini-ImageNet datasets, but the strategies can be applied to other small datasets as well.

## Datasets

We provide an easy script to test the data loading and preprocessing. You can run the following command to download the datasets and test the data loading and preprocessing.

```bash
torchrun bin/data_profile.py --data_set cifar10 --data_path ../data/  --gin SimpleAugmentation.img_size=32 --export outputs/cifar10.txt

torchrun bin/data_profile.py --data_set imnet --data_path ../data/miniImagenet/ --gin SimpleAugmentation.img_size=64 --export outputs/imnet.txt 
```

**CIFAR10** is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.
**IMNET64** is a variant of ImageNet with downsampled images. Here, we use the mini-ImageNet dataset with 64x64 images, which can be found at [https://www.image-net.org/download-images.php]. The dataset contains 1000 classes and 1,281,167 images.

## Models

### SimCLR 
SimCLR is a simple contrastive learning framework that learns representations by maximizing agreement between differently augmented views of the same data sample. We provide a simple implementation of SimCLR in the `models/simclr.py` file. You can run the following command to train the SimCLR model on CIFAR10.

```bash
WANDB_NAME=simclr_cifar10 torchrun main_pretrain.py --data_set cifar10 --data_path ../data/  --batch_size 128 --epochs=100 --warmup_epochs=10 --ckpt_freq 100 --opt lion --blr=1e-4  --cfgs configs/cifar.gin configs/vitt.gin --gin build_model.model_fn=@SimCLR SimCLR.embed_dim=192
```


### MAE
Masked autoencoder is a simple pixel reconstruction model that learns to reconstruct the input image from the masked image. We provide a simple implementation of MAE in the `models/mae.py` file. You can run the following command to train the MAE model on CIFAR10.

```bash
WANDB_NAME=mae_cifar10 torchrun main_pretrain.py --data_set cifar10 --data_path ../data/  --batch_size 512 --epochs=100 --warmup_epochs=10 --ckpt_freq 100 --opt lion --blr=1e-4  --cfgs configs/cifar.gin --gin build_model.model_fn=@mae_tiny build_model.patch_size=4 build_model.img_size=32
```