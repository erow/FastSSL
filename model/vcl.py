"""
Reference: https://github.com/google-research/simclr
FFCV_DEFAULT_CACHE_PROCESS=1 WANDB_NAME=VCL torchrun --nproc_per_node=8  main_pretrain_ffcv.py --data_path $train100_path --gin build_model.model_fn=@VCL MultiviewPipeline.img_size=112 MultiviewPipeline.local_crops_number=0 --multiview  --batch_size 256 --epochs=400 --ckpt_freq 50 --online_prob --weight_decay=1e-4  --output_dir outputs/vcl
# Note
"""
import numpy as np
import torch
from torch import nn
import gin
from layers.backbone import create_backbone
from layers.operation import build_head, concat_all_gather
import torch.nn.functional as F

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

class CenterHead(nn.Module):
    def __init__(self, dim, num_clusters):
        super(CenterHead, self).__init__()
        self.centroid = nn.Parameter(torch.randn(dim))
        self.centroid = nn.utils.weight_norm(nn.Linear(dim, num_clusters, bias=False))
        self.centroid.weight_g.data.fill_(1)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        logits = torch.einsum('nc,mc->nm', [x, self.centroid])
        return logits

@gin.configurable
class VCL(nn.Module):
    def __init__(self, 
                 beta=0,
                 embed_dim=2048,
                 out_dim=256,
                 mlp_dim=2048,):
        super(VCL, self).__init__()        
        self.beta=beta
        self.embed_dim = embed_dim

        backbone = create_backbone()
        self.backbone = backbone
        self.projector = build_head(2,embed_dim,mlp_dim,out_dim, last_norm='none')
        self.predictor = build_head(2,out_dim,mlp_dim,out_dim*2, last_norm='none')
    
    @torch.no_grad()
    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        x = self.backbone(x)
        return dict(latent=x)

    def forward(self, samples, **kwargs):
        self.log = {}
        x1,x2 = samples[:2]
        local_x = samples[2:]
        
            
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        mu, logvar = self.predictor(z1).chunk(2, dim=-1)
        # fix variance
        logvar = torch.zeros_like(logvar)

        # q(yz)
        rank = torch.distributed.get_rank()
        B, D = mu.shape

        # avoid the centroids collapse
        # q(z|y) = H(z,y) - H(y) = q(zy) / (sum_z q(zy)): B x (B N)
        tz = concat_all_gather(z2)
        log_zy = -0.5 * (logvar.unsqueeze(1) + (tz.unsqueeze(0) - mu.unsqueeze(1)).pow(2) / logvar.exp().unsqueeze(1)).sum(dim=-1)

        log_zcy = torch.diagonal(log_zy, offset=rank*B)
        h_zcy = (torch.logsumexp(log_zy, dim=1) - log_zcy).mean()
        

        # minimize the distance between features and centroids
        # q(y|z) = q(zy)/ (sum_y q(zy)): B x (B N)
        tmu = concat_all_gather(mu.contiguous())
        tlogvar = concat_all_gather(logvar.contiguous())
        log_yz = -0.5 * (tlogvar.unsqueeze(0) + (tmu.unsqueeze(0) - z2.unsqueeze(1)).pow(2) / tlogvar.exp().unsqueeze(0)).sum(dim=-1)

        log_ycz = torch.diagonal(log_yz, offset=rank*B)
        h_ycz = (torch.logsumexp(log_yz, dim=1) - log_ycz).mean()


        kl_loss = kl_normal_loss(mu, logvar).sum()
        loss = h_zcy + h_ycz + kl_loss * self.beta
        self.log['h_zcy'] = h_zcy.item()
        self.log['h_ycz'] = h_ycz.item()
        self.log['kl_loss'] = kl_loss.item()
        return loss, self.log