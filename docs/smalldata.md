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

## Pretraining Models

### SimCLR 
SimCLR is a simple contrastive learning framework that learns representations by maximizing agreement between differently augmented views of the same data sample. We provide a simple implementation of SimCLR in the `models/simclr.py` file. You can run the following command to train the SimCLR model on CIFAR10.

```bash
WANDB_NAME=simclr-cifar10 torchrun main_pretrain.py --data_set cifar10 --data_path ../data/  --batch_size 512 --epochs=200 --warmup_epochs=10 --ckpt_freq 100 --opt lion --blr=1e-4  --cfgs configs/cifar.gin configs/vitt.gin --gin build_model.model_fn=@SimCLR SimCLR.embed_dim=192
```

### MoCo
Momentum contrastive learning is a simple contrastive learning framework that utilizes a momentum encoder to stabilize the training. We provide a simple implementation of MoCo in the `models/moco.py` file. You can run the following command to train the MoCo model on CIFAR10.

```bash
WANDB_NAME=moco-cifar10 torchrun --master_port=12387  main_pretrain_ema.py --data_set cifar10 --data_path ../data/  --batch_size 512 --epochs=200 --warmup_epochs=10 --ckpt_freq 100 --opt lion --blr=1e-4  --cfgs configs/cifar.gin configs/vitt.gin --gin build_model.model_fn=@MoCo MoCo.embed_dim=192 MoCo.mlp_dim=512 MoCo.out_dim=128
```

### DINO
DINO is a self-distillation framework with momentum encoder that learns representations by maximizing agreement between differently augmented views of the same data sample. We provide a simple implementation of DINO in the `models/dino.py` file. You can run the following command to train the DINO model on CIFAR10.

```bash
WANDB_NAME=dino-cifar10 torchrun main_pretrain_ema.py --data_set cifar10 --data_path ../data/  --batch_size 512 --epochs=200 --warmup_epochs=10 --ckpt_freq 100 --opt lion --blr=1e-4  --cfgs configs/cifar.gin configs/vitt.gin --gin build_model.model_fn=@DINO DINO.embed_dim=192 DINO.out_dim=1024 -m 0.996
```

### SimSiam
SimSiam is a negative free self-supervised learning framework that learns representations by maximizing agreement between differently augmented views of the same data sample. We provide a simple implementation of SimSiam in the `models/simsiam.py` file. You can run the following command to train the SimSiam model on CIFAR10.

```bash
WANDB_NAME=simsiam-cifar10 torchrun main_pretrain.py --data_set cifar10 --data_path ../data/  --batch_size 512 --epochs=200 --warmup_epochs=10 --ckpt_freq 100 --opt lion --blr=1e-4  --cfgs configs/cifar.gin configs/vitt.gin --gin build_model.model_fn=@SimSiam SimSiam.embed_dim=192 SimSiam.proj_dim=192 SimSiam.mlp_dim=96
```


### MAE
Masked autoencoder is a simple pixel reconstruction model that learns to reconstruct the input image from the masked image. We provide a simple implementation of MAE in the `models/mae.py` file. You can run the following command to train the MAE model on CIFAR10.

```bash
WANDB_NAME=mae-cifar10 torchrun main_pretrain.py --data_set cifar10 --data_path ../data/  --batch_size 512 --epochs=200 --warmup_epochs=10 --ckpt_freq 100 --opt lion --blr=1e-4  --cfgs configs/cifar.gin --gin build_dataset.transform_fn=@SimpleAugmentation SimpleAugmentation.img_size=32 build_model.model_fn=@mae_tiny build_model.patch_size=4 build_model.img_size=32


WANDB_NAME=amae-cifar10 torchrun main_pretrain.py --data_set cifar10 --data_path ../data/  --batch_size 512 --epochs=200 --warmup_epochs=10 --ckpt_freq 100 --opt lion --blr=1e-4  --cfgs configs/cifar.gin --gin build_dataset.transform_fn=@SimpleAugmentation SimpleAugmentation.img_size=32 build_model.model_fn=@amae_tiny build_model.patch_size=4 build_model.img_size=32 build_model.decoder_patch_size=2 build_model.sigma=20 
```

### AIM

```bash
WANDB_NAME=aim-cifar10 torchrun --master_port 20953 --nproc_per_node=4 main_pretrain.py --data_set cifar10 --data_path ../data/ --batch_size 512 --epochs=200 --warmup_epochs=10 --ckpt_freq 100 --opt lion --blr=1e-4 --cfgs configs/cifar.gin --gin build_dataset.transform_fn=@SimpleAugmentation SimpleAugmentation.img_size=32 build_model.model_fn=@aim_tiny 
```

## Evaluation

finetune
```bash
CUDA_VISIBLE_DEVICES=6 WANDB_NAME=simclr-vitt python finetune.py --vit_mlp_ratio=4 --opt lion --lr=5e-5  -w ../FastSSL/outputs/simclr-cifar10-s0/weights.pth --prefix='backbone.(.*)'  


CUDA_VISIBLE_DEVICES=0 WANDB_NAME=mae-vitt torchrun finetune.py --vit_mlp_ratio=4 --opt lion --lr 1e-4 -w ../FastSSL/outputs/mae-cifar10/weights.pth 

vitrun eval_cls.py --data_set CIFAR10 --data_location ../data/ --gin build_model.model_name=\'vit_tiny_patch16_224\'  build_model.patch_size=4 build_model.img_size=32 --input_size=32 --prefix='backbone.(.*)'


```

KNN for fast evaluation
```bash
vitrun eval_knn.py --data_set CIFAR10 --data_location ../data/ --gin build_model.model_name=\'vit_tiny_patch16_224\' build_model.num_heads=12 build_model.patch_size=4 build_model.global_pool=\'avg\' build_model.img_size=32 --input_size=32  --prefix='<regex>' -w '<weight>'
```
