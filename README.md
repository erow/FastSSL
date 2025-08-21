# Toward Training Self-supervised Models with Limited Budget

This repository focuses on enabling efficient training for self-supervised learning (SSL). Often referred to as the "dark matter" of intelligence, SSL empowers AI systems to learn without supervision, drawing insights from their environments in ways reminiscent of human learning. While numerous advanced SSL algorithms have been proposed, many achieving state-of-the-art (SOTA) results, their adoption is often hindered by prohibitively high training costs. This limitation stifles innovation from academia and individual researchers. Designed to be beginner-friendly, this repository allows users to reproduce SSL algorithms and perform fast validation for new ideas. Here are key features:
- Efficient data loading with [ffcv](https://github.com/erow/ffcv).
- Flexible configuration with [gin-config](docs/config.md).
- A collection of SSL algorithms.
- Evaluation with [vitookit](https://github.com/erow/vitookit).
- All models are available at [WANDB](https://wandb.ai/erow/FastSSL).
- A [guideline](docs/smalldata.md) of training SSL models on CIFAR10 in a few minutes!.

# Environment Setup

Create a new environment with conda or micromamba:
```bash
conda create -y -n FastSSL python=3.10 cupy pkg-config 'libjpeg-turbo=3.0.0' opencv numba  pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge 
conda activate FastSSL
pip install -r requirements.txt
```
Or, you can use a docker image to ensure everything is the same with mine from [Github Package](https://github.com/erow/aisurrey-docker)
```
docker pull ghcr.io/erow/aisurrey-docker:sha256-d835a01e444257345d78c95cec157eb604a73935f70f9e7928cdd08d97411fa7.sig
```

# Usage

## torchrun

To train a MAE, you can run the following command
```bash 
torchrun --nproc_per_node 8 main_pretrain.py  --data_path=${train_path} --data_set=ffcv \
    --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --opt_betas 0.9 0.95 --batch_size 512\
    --cfgs configs/mae_ffcv.gin --gin build_model.model_fn=@base/MaskedAutoencoderViT   --ckpt_freq=100 --output_dir outputs/IN1K_base 
```
Optional arguments: `--compile` to compile the model, `--ckpt_freq` to save checkpoints every `ckpt_freq` epochs, `--online_prob` to evaluate the linear classifier during training.


## HPC

The original settings for [ViT-Large](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md) are bs=4096, epochs=800 ~42h in 64 V100 GPUs.

```bash
WANDB_NAME=mae_1k python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    -p gpu --ngpus 8 --nodes 8 \
    --batch_size 64 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --cfgs configs/mae_ffcv.gin --gin build_model.model_fn=@base/MaskedAutoencoderViT build_dataset.transform_fn=@SimplePipeline \
    --data_path=${train_path} --data_set=ffcv 
```

# Cite Me!

```bib
@misc{wu2024dailymaepretrainingmaskedautoencoders,
      title={DailyMAE: Towards Pretraining Masked Autoencoders in One Day}, 
      author={Jiantao Wu and Shentong Mo and Sara Atito and Zhenhua Feng and Josef Kittler and Muhammad Awais},
      year={2024},
      eprint={2404.00509},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.00509}, 
}
```