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

from os import getpid
from psutil import Process, net_io_counters
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from main_pretrain import *
from torch.profiler import profile, record_function, ProfilerActivity, schedule


class ramqdm(tqdm):
    """tqdm progress bar that reports RAM usage with each update"""
    _empty_desc = "using ? GB RAM; ?  CPU ? IO"
    _desc = "{:.2f} GB RAM; {:.2f} % CPU {:.2f} MB IO"
    _GB = 10**9
    """"""
    def __init__(self, *args, **kwargs):
        """Override desc and get reference to current process"""
        if "desc" in kwargs:
            # prepend desc to the reporter mask:
            self._empty_desc = kwargs["desc"] + " " + self._empty_desc
            self._desc = kwargs["desc"] + " " + self._desc
            del kwargs["desc"]
        else:
            # nothing to prepend, reporter mask is at start of sentence:
            self._empty_desc = self._empty_desc.capitalize()
            self._desc = self._desc.capitalize()
        super().__init__(*args, desc=self._empty_desc, **kwargs)
        self._process = Process(getpid())
        self.metrics = []
    """"""
    def update(self, n=1):
        """Calculate RAM usage and update progress bar"""
        rss = self._process.memory_info().rss
        ps = self._process.cpu_percent()
        io_counters = self._process.io_counters().read_bytes
        # net_io = net_io_counters().bytes_recv
        # io_counters += net_io
        
        current_desc = self._desc.format(rss/self._GB, ps, io_counters/1e6) + f" pid {getpid()} "
        self.set_description(current_desc)
        self.metrics.append({'mem':rss/self._GB, 'cpu':ps, 'io':io_counters/1e6})
        super().update(n)
    
    def summary(self):
        res = {}
        for key in self.metrics[0].keys():
            res[key] = np.mean([i[key] for i in self.metrics])
        return res
    
def backward_hook_wrapper(module, details=None):
    
    # define register_full_backward_pre_hook function
    def bwd_pre_hook_print(self, output):
        message = f'before backward of {module.__class__.__qualname__}'
        if details:
            message = f'{message}: {details}'
        with torch.profiler.record_function(message):
            return output

    # define register_full_backward_hook function
    def bwd_hook_print(self, input, output):
        message = f'after backward of {module.__class__.__qualname__}'
        if details:
            message = f'{message}: {details}'
        with torch.profiler.record_function(message):
            return input
    print(f"Register backward hook for {module.__class__.__qualname__}")
    # register hooks
    module.register_full_backward_pre_hook(bwd_pre_hook_print)
    module.register_full_backward_hook(bwd_hook_print)
    return module

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

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
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        data_loader_train = dataset_train
        print("Memory Manager = %s" % str(data_loader_train.memory_manager)) 
    print("data set : ", dataset_train)
    # define the model
    model = build_model(args)
    model.to(device)

    torch.compile(model)
    
    model_without_ddp = model

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
    loss_scaler = NativeScaler()
    
    print("Preload data")
    
    for _ in ramqdm(data_loader_train): 
        pass
    model.train()
    print(f"Start profiling for {args.num_samples} samples.")
    
    scaler = torch.cuda.amp.GradScaler()
    
    
    ## Profiling
    if args.no_profile:
        
        for _ in range(3):
            print("Start training one epoch.")
            l = ramqdm(data_loader_train)
            start = time.time()
            num_samples = 0
            for data_iter_step, data in enumerate(l):
                samples,y = data
                num_samples+=len(samples)
                with torch.cuda.amp.autocast():
                    loss = model(samples,epoch=0)
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                
            end = time.time()
            res = l.summary()
            res.update(args.__dict__)
            res['runtime'] = end-start
            res['throughput'] = float(num_samples)/(end-start)
            
            print(f"throughput : {res['throughput']} ")
            with open(os.path.join(args.output_dir, f"train_one_epoch-{global_rank}.json"), "a+") as file:
                file.write(json.dumps(res)+"\n")
    else:
        my_schedule = schedule(
            skip_first=100,
            wait=5,
            warmup=5,
            active=10)
        print_freq = 10
        optimizer.zero_grad()
        n_samples = 0
        
        print("Start profiling.")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,use_cuda=True,schedule=my_schedule,with_stack=True) as prof:
            metric_logger = misc.MetricLogger(delimiter="  ")
            for data_iter_step, data in enumerate(metric_logger.log_every(data_loader_train, print_freq, "")):
                with record_function('forward'):
                    if args.data_set == "ffcv":
                        samples = data[0]
                    else:
                        (samples, _) = data
                        samples = samples.cuda(non_blocking=True)

                    with torch.cuda.amp.autocast():
                        loss = model(samples,epoch=0)
                with record_function('backward'):
                    scaler.scale(loss).backward()
                
                with record_function('opt'):
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                
                n_samples +=len(samples)
                if n_samples >=args.num_samples: 
                    prof.step()
                    n_samples = 0 
                
                if prof.step_num >= 120: break
        
        print(prof.key_averages(group_by_stack_n=3).table(sort_by="self_cuda_time_total", row_limit=10))
        
        prof.export_chrome_trace(os.path.join(args.output_dir, f"profile-{global_rank}.json"))

if __name__ == '__main__':
    from util.helper import  aug_parse
    parser = get_args_parser()
    parser.add_argument("-n", "--num_samples", type=int, default=512, help="number of samples to record one step for profile.")
    parser.add_argument("--no_profile",default=False,action="store_true",help="whether to profile the model.")
    args = aug_parse(parser)
    assert args.num_samples > 0, "num_samples should be larger than 0."
    assert args.num_samples % args.batch_size == 0, "num_samples should be divisible by batch_size."
    
    if args.output_dir:
        output_dir=Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
