from PIL import Image # a trick to solve loading lib problem
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn

from ffcv import Loader
from ffcv.loader import OrderOption

import util.misc as misc
from util.misc import load_pretrained_weights
from dataset import ffcv_transform 
from dataset.multiloader import MultiLoader
from main_pretrain import get_args_parser, post_args, train, build_model

def main(args):
    misc.init_distributed_mode(args)
    post_args(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # build data loader
    order = OrderOption.RANDOM if args.distributed else OrderOption.QUASI_RANDOM
    if args.multiview:
        data_loader_train =  MultiLoader(args.data_path, pipelines=ffcv_transform.MultiviewPipeline(),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            batches_ahead=4, 
                            order=order, distributed=args.distributed,seed=args.seed)
    else:
        data_loader_train = Loader(args.data_path, pipelines=ffcv_transform.SimplePipeline(),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            batches_ahead=4, 
                            order=order, distributed=args.distributed,seed=args.seed)
    args.data_set = 'ffcv'
    # build the model
    model = build_model(args)
    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights)
    if args.compile:
        model = torch.compile(model)

    train(args,data_loader_train,model)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
