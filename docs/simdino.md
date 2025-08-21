```bash

main_pretrain_ema.py --data_set ffcv  --data_path ../data/ffcv/IN1K_train_smart.ffcv  --batch_size 128 --epochs=100 --warmup_epochs=10 --ckpt_freq 100 --cfgs configs/vitb.gin --gin build_model.model_fn=@SimDINO  SimDINO.embed_dim=768 build_dataset.transform_fn=@MultiviewPipeline MultiviewPipeline.local_crops_number=8
```