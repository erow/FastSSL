"""
Reference: https://github.com/facebookresearch/dino

# Keypoints
- Center: the key to avoiding collapse.
- DINO head: WeightNorm is applied at the end and the weight_g (magnitude) is fixed. Therefore, it only optimizes the direction, which equals to L2-normalization. In addition, BN is removed. L2-normalization bottleneck stabilizes the training of DINO with deep projection head.
- output dimension: large output dimensionality improves the performance. 65536 is the best.


# Result:


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import gin
from timm.models import create_model
import timm
import numpy as np
from timm.utils import ModelEmaV2
from timm.layers import trunc_normal_

@gin.configurable
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    
@gin.configurable
class DINO(nn.Module):
    def __init__(self, backbone_name: str='vit_small_patch16_224', 
                 dim=4096,      
                 m=0.996,
                 teacher_temp=0.05, student_temp=0.1,
                 center_momentum=0.9):
        """
        dim: feature dimension (default: 4096)
        teacher_temp: softmax temperature for teacher. Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend  starting with the default value of 0.04 and increase this slightly if needed.
        student_temp: 
        """
        super().__init__()
        self.embed_dim = dim
        self.m = m
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, dim))
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # build encoders
        backbone = create_model(backbone_name,num_classes=0)
        in_dim = backbone(torch.randn(1, 3, 224, 224)).shape[-1]
        projector = DINOHead(in_dim, dim)
        self.student = nn.Sequential(backbone,projector)

        # _teacher = timm.utils.ModelEmaV2(self.student,m)
        _teacher = nn.Sequential(create_model(backbone_name,num_classes=0),DINOHead(in_dim, dim))
        _teacher.requires_grad_(False)
        self._teacher = _teacher

    @torch.no_grad()
    def teacher(self,x):
        return self._teacher(x)
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center /  dist.get_world_size()

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    @torch.no_grad()
    def update(self):
        for ema_v, model_v in zip(self._teacher.state_dict().values(), self.student.state_dict().values()):
            ema_v.copy_(self.m * model_v + (1.0 - self.m) * ema_v)

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
        q1 = self.student(x1)/self.student_temp
        q2 = self.student(x2)/self.student_temp

        k1 = self.teacher(x1)
        pk1 = F.softmax((k1 - self.center)/self.teacher_temp,dim=-1)
        k2 = self.teacher(x2)
        pk2 = F.softmax((k2 - self.center)/self.teacher_temp,dim=-1)

        loss = (
            torch.sum(- pk1 * F.log_softmax(q2,dim=1),-1) + 
            torch.sum(- pk2 * F.log_softmax(q1,dim=1),-1)
        ).mean()/2

        loss_local = 0
        for lx in local_x:
            lz = self.student(lx)/self.student_temp

            loss_local += (
                torch.sum(- pk1 * F.log_softmax(lz),-1) + 
                torch.sum(- pk2 * F.log_softmax(lz),-1)
            ).mean()/2

        self.log = {
            "loss":loss.item(),
            "loss_local":loss_local if loss_local==0 else loss_local.item() 
        }
        self.update_center(torch.cat([k1,k2]))
        return loss,self.log
    