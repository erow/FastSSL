"""
Reference: https://github.com/facebookresearch/moco-v3

# Note
- Projector: a MLP ending with BN. 2048-4096-256
- Predictor: a MLP ending without BN. 256-4096-256

## network
MoCo v3 consists of backbone $f(\cdot)$, projector $f_q(\cdot)$, predictor $f_k(\cdot)$, and their momentum version, where projector is updated by EMA, predictor is learnable. 

q = f_q(f_k(f(x))), k =g_k(g(x)).

The projector is crutial for the performance improvement refer to SimCLR.

The predictor has no BN at the end.

SyncBN is also beneficial.

## momentum encoder

# Result:
| Model    | pretrain epochs | pretrain crops | linear acc |   |
|----------|:---------------:|:--------------:|:----------:|:-:|
| resnet50 | 100             | 2x224          | 68.9       |   |
| resnet50 | 300             | 2x224          | 72.8       |   |
| resnet50 | 1000            | 2x224          | 74.6       |   |
| ViT-Small| 300             | 2x224          | 73.2       |   |
| ViT-Base | 300             | 2x224          | 76.7       |   |

## Hyper parameter

| Architecture | LR | Opt | WD | Other |
| resnet|      0.6 | LARS| 1e-6|    |
| vit          |1.5e-4| AdamW | .1 | stop-grad-conv1
"""

import torch
import torch.nn as nn
import gin
from timm.models import create_model
import timm

from model.operation import build_mlp, contrastive_loss

@gin.configurable
class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, 
                 backbone: str='resnet50', 
                 out_dim=256, 
                 mlp_dim=4096, 
                 hidden_dim=2048,
                 T=0.2, m=0.99):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.m = m
        self.embed_dim = out_dim

        # build encoders
        backbone = create_model(backbone, num_classes=hidden_dim)
        projector = build_mlp(1,hidden_dim,mlp_dim,out_dim)
        self.student = nn.Sequential(backbone,
                                     nn.BatchNorm1d(hidden_dim),nn.ReLU(),
                                     projector)

        self._teacher = timm.utils.ModelEmaV2(self.student,m)
        self._teacher.requires_grad_(False)

        self.predictor = build_mlp(2,out_dim,mlp_dim,out_dim,False)

    @torch.no_grad()
    def teacher(self,x):
        return self._teacher.module(x)
    
    def update(self):
        self._teacher.update(self.student)

    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        x = self.teacher(x)
        return x

    def forward(self, imgs, **kwargs):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        x1, x2 = imgs[:2]
        local_x = imgs[2:]

        # compute features
        q1 = self.predictor(self.student(x1))
        q2 = self.predictor(self.student(x2))
        k1 = self.teacher(x1)
        k2 = self.teacher(x2)
        loss = ( contrastive_loss(q1, k2, self.T) + 
                contrastive_loss(q2, k1, self.T))/2

        loss_local = 0
        for lx in local_x:
            lz = self.student(lx)
            lp = self.predictor(lz)

            loss_local += (
                contrastive_loss(q1,lp, self.T) + 
                contrastive_loss(lp,k1, self.T) + 
                contrastive_loss(q2,lp, self.T) +
                contrastive_loss(lp,k2, self.T)  
            )/4

        self.log = {
            "loss":loss.item(),
            "loss_local":loss_local if loss_local==0 else loss_local.item() 
        }
        return loss,self.log