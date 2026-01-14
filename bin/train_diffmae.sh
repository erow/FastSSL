#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=72
#SBATCH --mem-per-gpu=110G
#SBATCH --output=outputs/log/%J_%t_log.out   # stdout
#SBATCH --error=outputs/log/%J_%t_log.err    # stderr (separate file)
export WANDB_NAME=diffmae_${MODEL_SIZE}_baseline
OUTPUT_DIR=${OUTPUT_DIR:-outputs}"/$WANDB_NAME"

scontrol update JobId=$SLURM_JOBID JobName=${WANDB_NAME:-svitrun}
free -mh
nvidia-smi

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400
export SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-$SLURM_GPUS_ON_NODE}
echo dist $RDZV_HOST:$RDZV_PORT rdzv_id=$SLURM_JOB_ID $SLURM_JOB_NUM_NODES x $SLURM_GPUS_PER_NODE


export WANDB_NOTES="$SLURM_JOB_ID"
export PYTHONPATH=.


# Default parameters for DiffMAE training
MODEL_SIZE=${1:-base}  # tiny, small, base, large, huge (default: base)
DATA_PATH=${FFCVTRAIN:-~/data/ffcv/IN100_train_500.ffcv}
EPOCHS=1600
BATCH_SIZE=$((4096 / (SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)))

# Additional arguments can be passed after the first 5 positional arguments
ADDITIONAL_ARGS=${@:2}

# Set model function based on model size
case $MODEL_SIZE in
    tiny)
        MODEL_FN="@tiny/diffmae_tiny"
        ;;
    small)
        MODEL_FN="@small/diffmae_small"
        ;;
    base)
        MODEL_FN="@base/diffmae_base"
        ;;
    large)
        MODEL_FN="@large/diffmae_large"
        ;;
    huge)
        MODEL_FN="@huge/diffmae_huge"
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Using base."
        MODEL_FN="@base/diffmae_base"
        ;;
esac

# Training command

srun python3 -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
   main_pretrain.py \
    --data_path=${DATA_PATH} \
    --data_set=ffcv \
    --epochs ${EPOCHS} \
    --warmup_epochs 40 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --batch_size ${BATCH_SIZE} \
    --output_dir ${OUTPUT_DIR} \
    --cfgs configs/diffmae_ffcv.gin \
    --gin build_model.model_fn=${MODEL_FN}  \
    --ckpt_freq=100 \
    ${ADDITIONAL_ARGS}

