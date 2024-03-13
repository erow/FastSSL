"""example usage:
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/
write_dataset train 500 0.50 90
write_path=$WRITE_DIR/train500_0.5_90.ffcv
echo "Writing ImageNet train dataset to ${write_path}"
python dataset/write_dataset.py \
    --cfg.dataset=imagenet \
    --cfg.split=train \
    --cfg.data_dir=$IMAGENET_DIR \
    --cfg.write_path=$write_path \
    --cfg.max_resolution=500 \
    --cfg.write_mode=proportion \
    --cfg.compress_probability=0.50 \
    --cfg.jpeg_quality=90
"""
from PIL import Image
from torch.utils.data import Subset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10, ImageFolder

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config
import cv2
import numpy as np

# hack resizer
# def resizer(image, target_resolution):
#     if target_resolution is None:
#         return image
#     original_size = np.array([image.shape[1], image.shape[0]])
#     ratio = target_resolution / original_size.min()
#     if ratio < 1:
#         new_size = (ratio * original_size).astype(int)
#         image = cv2.resize(image, tuple(new_size), interpolation=cv2.INTER_AREA)
#     return image
# from ffcv.fields import rgb_image
# rgb_image.resizer = resizer

Section('cfg', 'arguments to give the writer').params(
    dataset=Param(And(str, OneOf(['cifar', 'imagenet'])), 'Which dataset to write', default='imagenet'),
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    write_path=Param(str, 'Where to write the new dataset', required=True),
    write_mode=Param(str, 'Mode: raw, smart or jpg', required=False, default='smart'),
    max_resolution=Param(int, 'Max image side length. 0 any size.', required=False,default=0),
    num_workers=Param(int, 'Number of workers to use', default=16),
    chunk_size=Param(int, 'Chunk size for writing', default=100),
    jpeg_quality=Param(float, 'Quality of jpeg images', default=90),
    subset=Param(int, 'How many images to use (-1 for all)', default=-1),
    compress_probability=Param(float, 'compress probability', default=0.5)
)

@section('cfg')
@param('dataset')
@param('data_dir')
@param('write_path')
@param('max_resolution')
@param('num_workers')
@param('chunk_size')
@param('subset')
@param('jpeg_quality')
@param('write_mode')
@param('compress_probability')
def main(dataset, data_dir, write_path, max_resolution, num_workers,
         chunk_size, subset, jpeg_quality, write_mode,
         compress_probability):
    my_dataset = ImageFolder(root=data_dir)
    
    if subset > 0: my_dataset = Subset(my_dataset, range(subset))
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(write_mode=write_mode,
                               max_resolution=None if max_resolution==0 else max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
        'label': IntField(),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(my_dataset, chunksize=chunk_size)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    
    args=config.get().cfg
    assert args.write_path.endswith('.ffcv'), 'write_path must end with .ffcv'
    file=open(args.write_path.replace(".ffcv",".meta"), 'w')
    file.write(str(args.__dict__))
    main()
