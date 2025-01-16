# Configure

```text
usage: FastSSL pre-training script [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--accum_iter ACCUM_ITER] [--ckpt_freq CKPT_FREQ]
                                   [--no_wandb] [--dynamic_resolution] [--online_prob] [--compile] [-w PRETRAINED_WEIGHTS]
                                   [--opt OPTIMIZER] [--opt_eps EPSILON] [--opt_betas BETA [BETA ...]] [--clip_grad NORM] [--momentum M]
                                   [--weight_decay WEIGHT_DECAY] [--lr LR] [--blr LR] [--min_lr LR] [--warmup_epochs N]
                                   [--num_classes NUM_CLASSES] [--data_set DATA_SET] [--data_path DATA_PATH] [--output_dir OUTPUT_DIR]
                                   [--device DEVICE] [--seed SEED] [--resume RESUME] [--start_epoch N] [--num_workers NUM_WORKERS]
                                   [--pin_mem] [--no_pin_mem] [--world_size WORLD_SIZE] [--local-rank LOCAL_RANK] [--dist_on_itp]
                                   [--dist_url DIST_URL] [--no_resume] [--cfgs CFGS [CFGS ...]] [--gin GIN [GIN ...]]
```                                   
There are two types of arguments: program arguments and gin arguments. The program arguments basically controls the training process, such as epochs, optimizer, and output path. The program arguments are the essential parameters to launch the job and must be used. In contrast, the gin arguments are managed by [gin-config](https://github.com/google/gin-config) to configure the behaviour of models in a flexible way. The gin arguments are passed by `--gin k1=v1 [k2=v2 ...]`, or you can read the configure from files `--cfgs *.gin`. 


The library contains many models requiring different hyperparameters. To resolve the conflict of models (some of them may have the same hyperparameter name) and reduce cohesion between them, gin is the best tool as far as I know to change the parameters without modifying the main.

## build_model
`build_model.model_fn=@<model>` defines the entrance of which model would you like to build. You can further pass arguments to the model by adding `build_model.<k>=<v>`.

### create_backbone
`create_backbone` will cal `timm.create_model` to create the backbone network utilized in SimCLR and so on.

## build_transform
`build_transform.transform_fn=@<tf>` defines the entrance of data augmentation, which should be one of

- `@SimpleAugmentation` or `@SimplePipeline`: simple transforms used in MAE.
- `@DataAugmentationDINO` or `@MultiviewPipeline`: Multiview data augmentation