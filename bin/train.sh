MODEL=$1
PARAMS=${@:2}


# name=${MODEL}_rand
# WANDB_NAME=${name}  python submitit_pretrain.py --comment ${name} -p long -t 6000 --job_dir outputs/pretrain/${name} --data_path ~/data/edata/IN100K_rand.ffcv  $PARAMS --batch_size 128 --ckpt_freq 50 --online_prob --epochs 300

name=${MODEL}_c100
WANDB_NAME=${name} python submitit_pretrain.py --comment ${name} -p long -t 6000 --job_dir outputs/pretrain/${name} --data_path ~/data/ffcv/IN100_train_500.ffcv  $PARAMS --batch_size 128 --ckpt_freq 50 --online_prob --epochs 300

# name=${MODEL}_sas 
# WANDB_NAME=${name} python submitit_pretrain.py --comment ${name} -p long -t 6000 --job_dir outputs/pretrain/${name} --data_path ~/data/edata/SASIN100_resnet18.ffcv  $PARAMS --batch_size 128 --ckpt_freq 50 --online_prob --epochs 300

