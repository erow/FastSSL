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
import timm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from layers import build_model
from main_pretrain import get_args_parser
from util import misc

import timm.optim as optim_factory

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

from torch import nn
from layers.mae import Block
@gin.configurable()
class tf_base(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.block = Block(768,12,4,)
    
    def forward(self, x, **kwargs):
        # patchfy
        x = self.patch_embed(x)
        # flatten
        x = x.flatten(2).transpose(1, 2)
        
        for _ in range(12):
            x = self.block(x)
        loss = x.mean()
        return loss, None
        
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
    
    model = build_model.build_model(args)
    model.to(device)

    
    
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
    
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    # following timm: set wd as 0 for bias and norm layers
    if args.opt == 'lion':
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        optimizer = timm.optim.create_optimizer(args, param_groups)
    print("Preload data")
    

    model.train()

    scaler = torch.amp.GradScaler("cuda")
    
    
    
    samples = torch.randn(args.batch_size, 3, 224,224).cuda()
    if args.compile:
        model = torch.compile(model)
        # model = torch.jit.script(model)
    
    
    ## Profiling
    
    
    for data_iter_step in tqdm(range(100)):
        if data_iter_step==10:
            n_samples = 0
            start_time = time.time()
        elif data_iter_step>10:
            n_samples +=len(samples)
        
        with record_function('forward'):
            with torch.cuda.amp.autocast():
                loss,log = model(samples,epoch=0)
                
        with record_function('backward'):
            scaler.scale(loss).backward()
        
        with record_function('opt'):
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()       
    print('Throughput: %.1f samples/s' % (n_samples / (time.time() - start_time)))
    
    if not args.profile: return
    my_schedule = schedule(
        skip_first=20,
        wait=5,
        warmup=5,
        active=10)
    optimizer.zero_grad()
    
    print("Start profiling.")
   
    
    with profile(activities=[ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.output_dir),
        # profile_memory=True, with_stack=True,
        with_flops=True,
        schedule=my_schedule,
        ) as prof:
        for data_iter_step in tqdm(range(40)):
            with record_function('forward'):
                with torch.cuda.amp.autocast():
                    loss,log = model(samples,epoch=0)
                    n_samples +=len(samples)
            with record_function('backward'):
                scaler.scale(loss).backward()
            
            with record_function('opt'):
                scaler.step(optimizer)
                scaler.update()
                # torch.cuda.synchronize()
                        
            prof.step()            
    print('Throughput: %.1f samples/s' % (n_samples / (time.time() - start_time)))
    print(prof.key_averages(group_by_stack_n=1).table(sort_by="self_cuda_time_total", row_limit=10))
    
    # prof.export_chrome_trace(os.path.join(args.output_dir, f"profile.json"))

if __name__ == '__main__':
    from util.helper import  aug_parse
    parser = get_args_parser()
    parser.add_argument('--profile', action='store_true', help='profile',default=False)
    args = aug_parse(parser)
    
    if args.output_dir:
        output_dir=Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
