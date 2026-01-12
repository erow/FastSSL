# DiffMAE: Diffusion Models as Masked Autoencoders

## Overview

DiffMAE is a self-supervised learning method that combines the strengths of Masked Autoencoders (MAE) and Diffusion Models. Instead of directly predicting the masked patches, DiffMAE learns to reconstruct them through a diffusion process by iteratively denoising.

**Paper:** [DiffMAE: Diffusion Models as Masked Autoencoders](https://arxiv.org/abs/2304.03283)

## Key Features

1. **Diffusion-based Reconstruction**: Uses a diffusion decoder that learns to denoise gradually corrupted patches
2. **Flexible Architecture**: Supports various model sizes from Tiny (5.7M) to Huge (632M parameters)
3. **Better Representations**: The diffusion process provides richer training signals compared to direct pixel prediction
4. **Configurable Noise Schedules**: Supports linear, cosine, and quadratic noise schedules

## Architecture

### Components

1. **Encoder**: Standard Vision Transformer that processes visible (unmasked) patches
2. **Diffusion Decoder**: Transformer decoder with time-step embeddings that predicts noise
3. **Noise Scheduler**: Manages the forward diffusion process and noise addition

### Key Differences from MAE

- **MAE**: Encoder → Decoder → Direct pixel prediction
- **DiffMAE**: Encoder → Diffusion Decoder → Noise prediction at timestep t

## Model Sizes

| Model | Parameters | Encoder Dim | Encoder Depth | Decoder Dim | Decoder Depth |
|-------|-----------|-------------|---------------|-------------|---------------|
| Tiny  | 5.7M      | 192         | 12            | 96          | 4             |
| Small | 22M       | 384         | 12            | 192         | 4             |
| Base  | 86M       | 768         | 12            | 384         | 8             |
| Large | 304M      | 1024        | 24            | 512         | 8             |
| Huge  | 632M      | 1280        | 32            | 640         | 8             |

## Training

### Quick Start

#### Using FFCV Dataset

```bash
# Train DiffMAE-Base on ImageNet
torchrun --nproc_per_node=8 main_pretrain.py \
    --data_path=$FFCVTRAIN \
    --data_set=ffcv \
    --epochs 1600 \
    --warmup_epochs 40 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --batch_size 256 \
    --gin configs/diffmae_ffcv.gin \
    --ckpt_freq=100
```

#### Using ImageFolder Dataset

```bash
# Train DiffMAE-Base on ImageNet
torchrun --nproc_per_node=8 main_pretrain.py \
    --data_path=/path/to/imagenet/train \
    --data_set=imagefolder \
    --epochs 1600 \
    --warmup_epochs 40 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --batch_size 256 \
    --gin build_model.model_fn=@diffmae_base \
    --gin build_dataset.transform_fn=@SimplePipeline \
    --ckpt_freq=100
```

### Training Different Model Sizes

```bash
# Tiny model (faster training, good for small datasets)
torchrun --nproc_per_node=8 main_pretrain.py \
    --batch_size 512 \
    --gin build_model.model_fn=@diffmae_tiny \
    [other args...]

# Small model
torchrun --nproc_per_node=8 main_pretrain.py \
    --batch_size 512 \
    --gin build_model.model_fn=@diffmae_small \
    [other args...]

# Base model (recommended)
torchrun --nproc_per_node=8 main_pretrain.py \
    --batch_size 256 \
    --gin build_model.model_fn=@diffmae_base \
    [other args...]

# Large model
torchrun --nproc_per_node=8 main_pretrain.py \
    --batch_size 128 \
    --gin build_model.model_fn=@diffmae_large \
    [other args...]

# Huge model (requires significant compute)
torchrun --nproc_per_node=8 main_pretrain.py \
    --batch_size 64 \
    --gin build_model.model_fn=@diffmae_huge \
    [other args...]
```

### Hyperparameters

#### Recommended Settings (ImageNet-1K)

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Epochs | 1600 | Total training epochs (DiffMAE benefits from longer training) |
| Warmup Epochs | 40 | Learning rate warmup period |
| Base LR | 1.5e-4 | Base learning rate (scaled by batch size) |
| Weight Decay | 0.05 | AdamW weight decay |
| Optimizer | AdamW | Optimizer choice |
| Beta 1, Beta 2 | 0.9, 0.95 | AdamW beta parameters |
| Batch Size | 256 (Base) | Per-GPU batch size |
| Mask Ratio | 0.75 | Proportion of patches to mask |
| Diffusion Steps | 1000 | Number of diffusion timesteps |
| Noise Schedule | cosine | Type of noise schedule (cosine/linear/quadratic) |

#### Adjusting for Different Datasets

**Small datasets (CIFAR-10, CIFAR-100):**
```bash
--epochs 800 \
--batch_size 512 \
--blr 1e-4 \
--gin build_model.model_fn=@diffmae_tiny
```

**Medium datasets (ImageNet-100):**
```bash
--epochs 1000 \
--batch_size 256 \
--blr 1.5e-4 \
--gin build_model.model_fn=@diffmae_small
```

## Advanced Configuration

### Custom Mask Ratio

```bash
# Use 60% masking instead of default 75%
--gin diffmae_base.mask_ratio=0.6
```

### Different Noise Schedules

```bash
# Use linear noise schedule
--gin diffmae_base.diffusion_schedule='linear'

# Use quadratic noise schedule
--gin diffmae_base.diffusion_schedule='quadratic'

# Adjust number of diffusion timesteps
--gin diffmae_base.num_diffusion_timesteps=500
```

### Custom Model Architecture

```bash
# Customize encoder depth
--gin diffmae_base.depth=16

# Customize decoder depth
--gin diffmae_base.decoder_depth=6

# Adjust embedding dimension
--gin diffmae_base.embed_dim=512
```

## Using Pretrained Models

### Extract Representations

```python
import torch
from model.diffmae import diffmae_base

# Load pretrained model
model = diffmae_base()
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

# Extract representations
with torch.no_grad():
    imgs = torch.randn(4, 3, 224, 224)  # Example images
    features = model.representation(imgs)  # [4, embed_dim]
```

### Fine-tuning

The encoder can be used as a backbone for downstream tasks:

```python
from model.diffmae import diffmae_base

# Load pretrained encoder
model = diffmae_base()
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model'])

# Use encoder for classification
encoder = model  # or extract specific layers
classifier = torch.nn.Linear(model.embed_dim, num_classes)

# Fine-tune on downstream task
# ... training loop ...
```

## Monitoring Training

Key metrics to monitor:

1. **Loss**: Diffusion loss (MSE between predicted and actual noise)
2. **Learning Rate**: Should follow cosine schedule with warmup
3. **GPU Memory**: Monitor to adjust batch size if needed

Example W&B logging:
```bash
# Add W&B logging
--wandb_project your_project \
--wandb_entity your_entity \
--experiment_name diffmae_base_1600ep
```

## Comparison with MAE

| Aspect | MAE | DiffMAE |
|--------|-----|---------|
| Reconstruction Target | Direct pixels | Noise at timestep t |
| Decoder Complexity | Simpler | More complex (with time embedding) |
| Training Time | Faster | Slower (due to diffusion process) |
| Representation Quality | Good | Better (richer training signal) |
| Downstream Performance | Strong | Potentially stronger |

## Tips and Best Practices

1. **Longer Training**: DiffMAE benefits from longer training (1600 epochs) compared to MAE (800 epochs)
2. **Batch Size**: Use larger batch sizes when possible for better convergence
3. **Mask Ratio**: 0.75 works well, but try 0.6-0.8 range for your dataset
4. **Noise Schedule**: Cosine schedule generally works best
5. **Memory**: DiffMAE requires more memory than MAE due to diffusion decoder

## Troubleshooting

### Out of Memory

- Reduce batch size: `--batch_size 128`
- Use smaller model: `@diffmae_small` instead of `@diffmae_base`
- Enable gradient checkpointing (if implemented)

### Slow Training

- Reduce diffusion timesteps: `--gin diffmae_base.num_diffusion_timesteps=500`
- Use fewer decoder layers: `--gin diffmae_base.decoder_depth=4`
- Use mixed precision training (usually enabled by default)

### Poor Performance

- Increase training epochs
- Try different mask ratios
- Adjust learning rate
- Check data augmentation pipeline

## References

- **Paper**: [DiffMAE: Diffusion Models as Masked Autoencoders](https://arxiv.org/abs/2304.03283)
- **MAE Paper**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- **DDPM Paper**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

## Citation

If you use DiffMAE in your research, please cite:

```bibtex
@article{diffmae2023,
  title={DiffMAE: Diffusion Models as Masked Autoencoders},
  author={Wei, Chen and Mangalam, Karttikeya and Huang, Po-Yao and Li, Yanghao and Fan, Haoqi and Xu, Hu and Wang, Huiyu and Xie, Cihang and Yuille, Alan and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2304.03283},
  year={2023}
}
```

