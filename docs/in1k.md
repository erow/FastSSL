
# MAE
```
WANDB_NAME=mae_base_e800 torchrun --nproc_per_node=8 main_pretrain.py --data_path=$FFCVTRAIN --compile --data_set=ffcv --epochs 800 --warmup_epochs 40 --opt adamw --blr 1.5e-4 --weight_decay 0.05 --batch_size 128 --gin build_model.model_fn=@mae_base build_dataset.transform_fn=@SimplePipeline --ckpt_freq=100 --output_dir outputs/IN1K/mae_base_e800_accum
```