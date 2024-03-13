# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path
import importlib
import submitit
from util.helper import aug_parse

def parse_args():
    # trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for evaluation")
    parser.add_argument("--module", default="vitookit.evaluation.eval_cls", type=str, help="Module to run")
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("-t", "--timeout", default=1440, type=int, help="Duration of the job")
    parser.add_argument("--mem", default=400, type=float, help="Memory to request")

    parser.add_argument("-p", "--partition", default="big", type=str, help="Partition where to submit")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    parser.add_argument( "--job_dir", default='',type=str,)
    parser.add_argument('--fast_dir',default='', help="The dictory of fast disk to load the datasets")
    
    args, known= parser.parse_known_args()
    return args


def get_shared_folder(root) -> Path:
    root = root.replace("%j", "shared")
    p = Path(root)
    os.makedirs(str(p), exist_ok=True)
    if Path(root).is_dir():
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(root):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(root)), exist_ok=True)
    init_file = get_shared_folder(root) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.module = importlib.import_module(args.module)
        
        ## reassing args
        parser = self.module.get_args_parser()
        module_args = aug_parse(parser)
        module_args.output_dir = args.job_dir
        module_args.dist_url = args.dist_url
        self.module_args = module_args

    def __call__(self):
        self._setup_gpu_args()
        
        # move the dataset to fast_dir
        fast_dir = self.args.fast_dir
        if fast_dir:
            import shutil
            for key,value in self.module_args.__dict__.items():
                if isinstance(value,str) and '.ffcv' in value:
                    os.makedirs(fast_dir, exist_ok=True)
                    # Copy the file
                    if self.module_args.rank==0:
                        new_path = shutil.copy(value, fast_dir)                    
                        print("Copying ", value, " to ", new_path)
                    else:
                        new_path = os.path.join(fast_dir, os.path.basename(value))
                        print("Waiting for rank 0 to copy ", value, " to ", new_path)
                    self.module_args.__dict__[key] = new_path
        self.module.main(self.module_args)

    def checkpoint(self):
        print("Checkpointing")
        import os
        import submitit
        job_env = submitit.JobEnvironment()
        print("Requeuing ", self.args, self.module_args)
        
        output_dir = self.module_args.output_dir
        
        checkpoint_file = os.path.join(output_dir, "checkpoint.pth")  
        self.args.dist_url = get_init_file(output_dir).as_uri()
        empty_trainer = type(self)(self.args)      
        if os.path.exists(checkpoint_file):
            empty_trainer.module_args.resume = checkpoint_file
        
        print("Requeueing with ", empty_trainer.module_args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        module_args = self.module_args
        job_env = submitit.JobEnvironment()
        output_dir = str(self.args.job_dir).replace("%j", str(job_env.job_id))
        module_args.output_dir = output_dir
        
        module_args.gpu = job_env.local_rank
        module_args.rank = job_env.global_rank
        module_args.world_size = job_env.num_tasks
        
        module_args.comment = f"Job {job_env.job_id} on {job_env.num_tasks} GPUs"
        
        import gin
        if not gin.config_is_locked():
            gin.parse_config_files_and_bindings(module_args.cfgs,module_args.gin)
        print("Setting up GPU args", module_args)
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    args.module = 'main_pretrain'
    if args.job_dir=='':
        args.job_dir = f"outputs/experiments/%j"
    args.job_dir = os.path.abspath(args.job_dir)
    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=args.mem,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="train")
    args.dist_url = get_init_file(args.job_dir).as_uri()
    print("args:", args)
    trainer = Trainer(args)
    job = executor.submit(trainer)
    
    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()