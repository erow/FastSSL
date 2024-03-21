# simclr
WANDB_NAME=simclr torchrun --nproc_per_node=2 main_pretrain.py --cfgs configs/simclr_if.gin  --data_set=IMNET --data_path $IMAGENET_DIR  --batch_size=1024   --online_prob --weight_decay=0.0001 --epochs=100 --blr 5e-4 --warmup_epochs 10 --ckpt_freq 10 --num_workers=30 --output_dir outputs/simclr

# mae
WANDB_NAME=mae torchrun --nproc_per_node=2 main_pretrain.py --cfgs configs/mae_if.gin  --data_set=IMNET --data_path $IMAGENET_DIR  --batch_size=1024 --online_prob --blr 1.5e-4 --weight_decay 0.05 --epochs 800 --warmup_epochs 40  --ckpt_freq 100  --num_workers=30 --output_dir outputs/mae 