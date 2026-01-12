from PIL import Image  # a trick to solve loading lib problem
import dataset.ffcv_transform

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import gin
import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from util import misc
import timm

from util.helper import aug_parse
from util.prob import LinearProb, build_representations_fn

assert timm.__version__ >= "0.6.12"  # version check
import importlib
import pkgutil

import timm.optim as optim_factory

from timm.layers import (convert_splitbn_model, convert_sync_batchnorm,
                         set_fast_norm)

import util.lr_sched as lr_sched

from layers.build_model import build_model
from dataset.build_dataset import build_dataset

from typing import Iterable
from util.dres import DynamicMasking

def get_args_parser():
    parser = argparse.ArgumentParser('Fast Self-supervised Learning', add_help=False)
    parser.add_argument('--daccum',default=False,action='store_true')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--ckpt_freq',type=int,default=50, help="frequency of saving checkpoint.")
    parser.add_argument('--backup', default=True, action='store_true', help="backup the model.")
    parser.add_argument('--no_backup',action='store_false', dest='backup')
    parser.add_argument("--no_wandb", default=False, action="store_true", help="Use wandb for logging.")
    parser.add_argument("--dynamic_resolution", default=False, action="store_true", help="Use dynamic resolution.")
    parser.add_argument("--prob", default=False, action='store_true')
    parser.add_argument("--line_prob", default=False, action='store_true', help="Use linear probability evaluation during training.")
    
    # Model parameters
    parser.add_argument("--compile", default=False, action="store_true", help="Compile the module or not.")
    parser.add_argument("-w", "--pretrained_weights", default=None, type=str, help="Path to pretrained weights.")

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_filter',default=False, action='store_true', 
                        help='If True, bias, norm layer parameters (all 1d params) will not have weight decay applied. Only used when model_or_params is a model and weight_decay > 0.')
    parser.add_argument('--opt_betas', default=(0.9,0.95), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
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
    parser.add_argument('--num_classes', default=1000, type=int,)
    parser.add_argument('--data_set', default='imnet', help='dataset name, one of {imnet, ffcv, cifar10}')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default=None, type=str,
                        help='validation dataset path for linear probing evaluation')

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
    parser.add_argument("--syncBN", default=False, action="store_true", help="Use Global batch normalization to sync all devices.")
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank','--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


                
def train_one_epoch(model, online_prob, 
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
        if args.data_set == "ffcv":
            samples = data[:-1]
            targets = data[-1]            
        else:
            samples, targets = data
        
        if isinstance(samples,list) or isinstance(samples,tuple):
            samples = [i.to(device, non_blocking=True) for i in samples]
            if len(samples)==1:
                samples = samples[0]
        else:
            samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).flatten()

        with torch.amp.autocast('cuda',dtype=torch.float16):
            loss, log = model(samples,targets=targets, epoch=epoch)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            torch.save(model.module, "nan_model.pt")
            sys.exit(1)

        loss /= accum_iter
        
        if (data_iter_step + 1) % accum_iter == 0:
            # update step 
            
            # we use a per iteration (instead of per epoch) lr scheduler
            lr = lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
            # gradient step
            norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=True)
            optimizer.zero_grad()
        
            if online_prob:
                prob_log = online_prob.step(samples, targets)
                log.update(prob_log)
            
            
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=lr)
            for k,v in log.items():
                metric_logger.update(**{k:v})
            
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('epoch_1000x',epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                log_writer.add_scalar('norm',norm,epoch_1000x)
                for k,v in log.items():
                    log_writer.add_scalar(f'{k}', v, epoch_1000x)
        else:
            norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=False)
        if args.device == 'cuda':
            torch.cuda.synchronize()

        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    misc.init_distributed_mode(args)
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    misc.init_seed(seed)
    cudnn.benchmark = True

    # build dataset
    dataset_train = build_dataset(args)
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    args.eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()    
    
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
            persistent_workers=True if args.num_workers > 0 else False,
        )
    else:
        data_loader_train = dataset_train
    
    # initialize model
    model = build_model(args)
    if args.pretrained_weights:
        misc.load_pretrained_weights(model, args.pretrained_weights)
    if args.compile:
        model = torch.compile(model)
    train(args, data_loader_train, model)


def train(args, data_loader_train,model):
    import torch
    device = torch.device(args.device)
    if args.output_dir:
        output_dir=Path(args.output_dir)

    global_rank = misc.get_rank()
    if global_rank == 0 and args.output_dir is not None:
        args.log_dir = os.path.join(args.output_dir, 'log')
        os.makedirs(args.log_dir, exist_ok=True)
        # log with wandb
        open(output_dir/"config.gin",'w').write(gin.operative_config_str(),)
        if not args.no_wandb:
            import wandb
            wandb.init(dir=args.log_dir,config=args.__dict__,sync_tensorboard=True,resume='allow', job_type='train')
            wandb.save(os.path.join(args.output_dir, 'config.gin'),base_path=args.output_dir)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.dynamic_resolution:
        import torch._dynamo
        from model.dres import DynamicMasking
        torch._dynamo.config.suppress_errors = True
        dres = DynamicMasking() 
    else:
        dres = None
    
    
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: %f M" % (num_params/1e6))

    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.eff_batch_size / 256 

    print("base lr: %.2e" % (args.lr * 256 / args.eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % args.eff_batch_size)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)     
    
    # if args.opt == 'adamw':
    #     optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9,0.95),weight_decay=args.weight_decay)   
    # elif args.opt == 'muon':
    #     from util.muon import Muon
    #     muon_params = []
    #     adamw_params = []
    #     no_weight_decay_list = []
    #     for name, p in model.named_parameters():
    #         if 'block' in name and p.ndim == 2:
    #             muon_params.append(p)
    #         elif p.ndim <= 1 or name.endswith(".bias"):
    #             no_weight_decay_list.append(p)
    #         else:
    #             adamw_params.append(p)
    #     print(f"Muon parameters: {len(muon_params)}, AdamW parameters: {len(adamw_params)}")
    #     # if args.opt_betas is None:
    #     #     args.opt_betas = (0.95,0.95)
    #     muon_params = [{"params": muon_params,'weight_decay':1}]
    #     adamw_params = [
    #         {'params': no_weight_decay_list, 'weight_decay': 0.},
    #         {'params': adamw_params, 'weight_decay': args.weight_decay},
    #         ]
    #     optimizer = Muon(lr=args.lr, wd=args.weight_decay, momentum=0.95,
    #                      muon_params=muon_params, adamw_params=adamw_params,
    #                      adamw_eps=args.opt_eps)

    # else:
    #     optimizer = timm.optim.create_optimizer(args, param_groups)
    
    optimizer = optim_factory.create_optimizer(args, model,filter_bias_and_bn=args.opt_filter)
    print('optimizer', optimizer)
    loss_scaler = misc.NativeScalerWithGradNormCount()

    if args.resume is None:
        if args.output_dir and os.path.isfile(os.path.join(args.output_dir, "checkpoint.pth")):
            args.resume = os.path.join(args.output_dir, "checkpoint.pth")
    
    if args.resume: print("Found checkpoint at %s" % args.resume)
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    
    start_time = time.time()
    if args.prob:
        from util.clustering import KmeansProb        
        online_prob = KmeansProb(model_without_ddp.representation,
            num_clusters=args.num_classes)
    elif args.line_prob:
        if args.val_data_path is None:
            raise ValueError("--val_data_path must be provided when using --line_prob")
        names, representations_fn = build_representations_fn(model_without_ddp)
        online_prob = LinearProb(args.val_data_path, names, representations_fn, num_classes=args.num_classes, device=device)
        print("Loaded LinearProb successfully:", online_prob)
    else:
        online_prob = None
        
    base_accum=args.accum_iter
        
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.data_set != "ffcv":
            data_loader_train.sampler.set_epoch(epoch)
        if dres:
            dres(model_without_ddp, data_loader_train,epoch,is_ffcv=args.data_set == "ffcv")

        if args.daccum:
            ## TODO: dynamically increasing accum_iter
            if epoch >= int(args.epochs*0.75):
                args.accum_iter = base_accum * 2
            elif epoch >= int(args.epochs*0.5) :
                args.accum_iter = base_accum * 2
            ##
        
        train_stats = train_one_epoch(
            model, online_prob,data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        

        if args.output_dir:
            if (epoch % args.ckpt_freq == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args, model=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
            if args.backup:
                misc.save_model(
                    args=args, model=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch,bac=False)

        log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        
        if args.output_dir and misc.is_main_process():
            if not args.no_wandb:
                # Convert any tensor values to Python scalars for wandb
                wandb_log_stats = {}
                for k, v in log_stats.items():
                    if isinstance(v, torch.Tensor):
                        wandb_log_stats[k] = v.item()
                    elif isinstance(v, (int, float)):
                        wandb_log_stats[k] = float(v)
                    else:
                        wandb_log_stats[k] = v
                wandb.log(wandb_log_stats, step=epoch)
                
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if args.output_dir and misc.is_main_process():
        weights = model_without_ddp.state_dict()
        weights = {k.replace("_orig_mod.",""): v.cpu() for k, v in weights.items()}
        torch.save(weights, os.path.join(args.output_dir, "weights.pth"))
        if not args.no_wandb:
            wandb.save(os.path.join(args.output_dir, "weights.pth"),base_path=args.output_dir)

if __name__ == '__main__':
    parser = get_args_parser()
    args = aug_parse(parser)
    main(args)
