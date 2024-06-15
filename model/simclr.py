"""
Reference: https://github.com/google-research/simclr

# Note

## Keypoints in SimCLR

- data augmentation: random cropping and random color distortion stand out.
- Global BN (SyncBN): This operation aggregates BN mean and variance over all devices during the training.
- Projector: a MLP with BN. By leveraging the nonlinear transformation g(·), more information can be formed and maintained in h. 2048-2048-256

# Result


"""
import torch
from torch import nn
import torchvision.transforms as transforms
import gin
import timm
from .operation import *

@gin.configurable
class SimCLR(nn.Module):
    def __init__(self, backbone='resnet50', 
                 out_dim=256,
                 hidden_dim=2048,
                 mlp_dim=2048, 
                 temperature=0.5):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        backbone = timm.create_model(backbone,pretrained=False,num_classes=hidden_dim)
        self.backbone = backbone
        self.embed_dim = hidden_dim
        self.projector = build_mlp(1, hidden_dim, mlp_dim, out_dim, False)

    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        x = self.backbone(x)
        return x

    def update(self):
        pass

    def forward(self, samples, **kwargs):
        x1,x2 = samples[:2]
        local_x = samples[2:]
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        
        loss = (contrastive_loss(z1,z2,self.temperature) + 
                contrastive_loss(z2,z1,self.temperature))/2

        loss_local = 0
        for lx in local_x:
            lz = self.backbone(lx)
            lp = self.projector(lz)

            loss_local += (
                contrastive_loss(z1,lp,self.temperature) + 
                contrastive_loss(lp,z1,self.temperature) + 
                contrastive_loss(z2,lp,self.temperature) +
                contrastive_loss(lp,z2,self.temperature)  
            )/4

        self.log = {
            "loss":loss.item(),
            "loss_local": 0 if loss_local==0 else loss_local.item()
        }

        return loss+ loss_local, self.log