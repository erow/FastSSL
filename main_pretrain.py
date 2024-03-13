# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from PIL import Image # a trick to solve loading lib problem
import argparse
import datetime
import json
import numpy as np
import os
import time
import math
import sys
import gin
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ >= "0.6.12"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler, load_pretrained_weights
import util.lr_sched as lr_sched

from model.build_model import build_model
from dataset.build_dataset import build_dataset
from model.dres import DynamicMasking
from dataset import ffcv_transform 

from typing import Iterable

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--ckpt_freq',type=int,default=10, help="frequency of saving checkpoint.")
    parser.add_argument("--no_wandb", default=False, action="store_true", help="Use wandb for logging.")
    parser.add_argument("--dynamic_resolution", default=False, action="store_true", help="Use dynamic resolution.")
    
    # Model parameters
    parser.add_argument('--online_prob', default=False, action="store_true", help="Use online prob.")
    parser.add_argument("--compile", default=False, action="store_true", help="Compile the module or not.")
    parser.add_argument("--flash_attn", default=False, action="store_true", help="Use flash attention.")
    parser.add_argument("-w", "--pretrained_weights", default=None, type=str, help="Path to pretrained weights.")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_set', default='ffcv', choices=['PZ1', 'IMNET', 'PZ2', 'ffcv'])
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank','--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


                
def train_one_epoch(model: torch.nn.Module,online_prob, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.data_set != "ffcv":
            samples, targets = data
            
        else:
            samples, targets = data
        
        if isinstance(samples,list) or isinstance(samples,tuple):
            samples = [i.to(device, non_blocking=True) for i in samples]
        else:
            samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
            
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        

        with torch.cuda.amp.autocast():
            loss = model(samples,epoch=epoch)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            
            if hasattr(model,"update"):
                model.update()
                
            if online_prob:
                acc = online_prob.train_one_step(samples, targets.flatten())
                metric_logger.update(acc=acc)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[-1]["lr"]
        metric_logger.update(lr=lr)
        
        

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('epoch_1000x',epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class OnlineProb():
    def __init__(self, model,lr=1e-3):
        self.model = model
        from torch import nn
        embd_dim = model.embed_dim
        self.classifer = nn.Sequential(
            nn.BatchNorm1d(embd_dim),
            nn.Linear(embd_dim, 1000),
        )
        self.optimizer = torch.optim.Adam(self.classifer.parameters(), lr=lr,)
        
    def train_one_step(self, samples, targets):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                features = self.model.representation(samples)
        logits = self.classifer(features)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return acc.item()
    

def main(args):
    misc.init_distributed_mode(args)

    if args.output_dir:
        output_dir=Path(args.output_dir)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    import torch
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(args)
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.data_set != "ffcv":
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        data_loader_train = dataset_train
    
    if args.dynamic_resolution:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        dres = DynamicMasking() 
    else:
        dres = None
    
    # define the model
    model = build_model(args)
    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights)
    if args.compile:
        model = torch.compile(model)
    
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: %f M" % (num_params/1e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.resume is None:
        if args.output_dir and os.path.isfile(os.path.join(args.output_dir, "checkpoint.pth")):
            args.resume = os.path.join(args.output_dir, "checkpoint.pth")
    
    if args.resume: print("Found checkpoint at %s" % args.resume)
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    
    if global_rank == 0 and args.output_dir is not None:
        args.log_dir = os.path.join(args.output_dir, 'log')
        os.makedirs(args.log_dir, exist_ok=True)
        # log with wandb
        open(output_dir/"config.gin",'w').write(gin.operative_config_str(),)
        if not args.no_wandb:
            import wandb
            wandb.init(dir=args.log_dir,config=args.__dict__,sync_tensorboard=True,resume=True, job_type='train')
            wandb.save(os.path.join(args.output_dir, 'config.gin'),base_path=args.output_dir)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    
    start_time = time.time()
    online_prob = OnlineProb(model_without_ddp) if args.online_prob else None
    if online_prob:
        online_prob.classifer.to(device)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.data_set != "ffcv":
            data_loader_train.sampler.set_epoch(epoch)
        if dres:
            dres(model_without_ddp, data_loader_train,epoch,is_ffcv=args.data_set == "ffcv")
                
        train_stats = train_one_epoch(
            model, online_prob,data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.ckpt_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        else:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=-1)

        log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if not args.no_wandb:
                wandb.log(log_stats)
                
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # save the weights
    def remove_prefix(text, prefix):
        if text.startswith(prefix):
            return text[len(prefix) :]
        return text
    if args.output_dir and misc.is_main_process():
        weights = model_without_ddp.state_dict()
        if args.compile:
            weights = {remove_prefix(k,"_orig_mod."): v.cpu() for k, v in weights.items()}
        torch.save(weights, os.path.join(args.output_dir, "weights.pth"))
        if not args.no_wandb:
            wandb.save(os.path.join(args.output_dir, "weights.pth"),base_path=args.output_dir)

if __name__ == '__main__':
    from util.helper import aug_parse
    parser = get_args_parser()
    args = aug_parse(parser)
    main(args)
