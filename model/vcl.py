"""
Reference: https://github.com/google-research/simclr
FFCV_DEFAULT_CACHE_PROCESS=1 WANDB_NAME=VCL torchrun --nproc_per_node=8  main_pretrain_ffcv.py --data_path $train100_path --gin build_model.model_fn=@VCL MultiviewPipeline.img_size=112 MultiviewPipeline.local_crops_number=0 --multiview  --batch_size 256 --epochs=400 --ckpt_freq 50 --online_prob --weight_decay=1e-4  --output_dir outputs/vcl
# Note
"""
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import gin
import timm
from .operation import *

def kl_normal_loss(mean, logvar, mean_dim=None):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, num_latent) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        num_latent)
    """
    if mean_dim is None:
        mean_dim = [0]
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=mean_dim)
    return latent_kl

@gin.configurable
class VCL(nn.Module):
    def __init__(self, backbone='resnet50', 
                 out_dim=256,
                 hidden_dim=2048,
                 mlp_dim=2048, 
                 temperature=0.15):
        super(VCL, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature), requires_grad=True)
        self.embed_dim = hidden_dim
        backbone = timm.create_model(backbone,pretrained=False,num_classes=hidden_dim)
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),nn.ReLU(),
            build_mlp(2, hidden_dim, mlp_dim, out_dim, False),
        )

    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        x = self.backbone(x)
        return x

    def update(self):
        pass

    def forward(self, samples, **kwargs):
        self.log = {}
        x1,x2 = samples[:2]
        local_x = samples[2:]
        with torch.no_grad():
            mu = self.projector(self.backbone(x1))
        z = self.projector(self.backbone(x2))
        sigma2 = (z-mu).square()

        KL_loss = z.square() - sigma2.log() - 1
        KL_loss = (KL_loss/2).mean()
        
        temperature = self.logit_scale.exp()
        
        logits = torch.einsum('nc,mc->nm', [z, mu]) * temperature
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long)).cuda()
        cl_loss =  nn.CrossEntropyLoss()(logits, labels)
        loss = cl_loss 
        self.log['loss'] = loss.item()
        self.log['kl_loss'] = KL_loss.item()
        self.log['cl_loss'] = cl_loss.item()
        self.log['temperature'] = temperature.item()
        self.log['sigma2'] = sigma2.mean().item()

        return loss, self.log