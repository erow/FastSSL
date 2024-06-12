import torch
from torch import nn
import torchvision.transforms as transforms
import gin
import timm
from .operation import *

@gin.configurable
class SimCLR(nn.Module):
    def __init__(self, backbone='resnet18', out_dim=2048,mlp_dim=512, temperature=0.5):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        self.backbone = timm.create_model(backbone,pretrained=False,num_classes=out_dim)
        self.embed_dim = out_dim
        self.projector = build_mlp(2, out_dim, mlp_dim, out_dim)

    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        x = self.backbone(x)
        return x

    def forward(self, samples, **kwargs):
        x1,x2 = samples[:2]
        local_x = samples[2:]
        z1 = self.projector(self.representation(x1))
        z2 = self.projector(self.representation(x2))
        
        loss = contrastive_loss(z1,z2) + contrastive_loss(z2,z1)

        loss_local = 0
        for lx in local_x:
            lz = self.backbone(lx)
            lp = self.projector(lz)

            loss_local += (
                contrastive_loss(z1,lp) + 
                contrastive_loss(lp,z1) + 
                contrastive_loss(z2,lp) +
                contrastive_loss(lp,z2)  
            )/4

        self.log = {
            "loss":loss.item(),
            "loss_local":loss_local.item()
        }

        return loss+ loss_local, self.log
        