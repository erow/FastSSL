"""
Reference: https://github.com/google-research/simclr

# Note

## Keypoints in SimCLR

- data augmentation: random cropping and random color distortion stand out.
- Global BN (SyncBN): This operation aggregates BN mean and variance over all devices during the training.
- Projector: a MLP with BN. By leveraging the nonlinear transformation g(·), more information can be formed and maintained in h. 2048-2048-256
- Batch size: it is crucial for improving performance. BS=4096 achieves good results.
- Epoch: Contrastive learning benefits (more) from larger batch sizes and longer training. At least 400 epochs.

# Result


"""
import torch
from torch import nn
import torchvision.transforms as transforms
import gin
import timm
from layers.operation import *
from layers.backbone import create_backbone

@gin.configurable
class SimCLR(nn.Module):
    def __init__(self,
                 out_dim=256,
                 embed_dim=2048,
                 mlp_dim=2048, 
                 temperature=0.1):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        self.out_dim = out_dim
        backbone = create_backbone()
        
        self.embed_dim = embed_dim
        self.backbone = backbone
        self.projector = build_head(2,embed_dim,mlp_dim,out_dim, False)

    @torch.no_grad()
    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        latent = self.backbone(x)
        proj = self.projector(latent)
        rep = dict(latent=latent,proj=proj)
        return rep

    def forward(self, samples, **kwargs):
        self.log = {}
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
        if loss_local>0:
            loss = loss + loss_local
            self.log['loss_local'] = loss_local.item()
        self.log['z@sim'] = F.cosine_similarity(z1,z2).mean().item()

        return loss, self.log



def dynamic_contrastive_loss(hidden1, hidden2, index=None, gamma=0.9, distributed=True):
    """
    paper: Provable stochastic optimization for global contrastive learning: Small batch does not harm performance
    reference: https://github.com/Optimization-AI/SogCLR/blob/PyTorch/sogclr/builder.py#L66
    """
    # Get (normalized) hidden1 and hidden2.
    hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
    batch_size = hidden1.shape[0]
    
    # Gather hidden1/hidden2 across replicas and create local labels.
    if distributed:  
        hidden1_large = torch.cat(all_gather_layer.apply(hidden1), dim=0) # why concat_all_gather()
        hidden2_large =  torch.cat(all_gather_layer.apply(hidden2), dim=0)
        enlarged_batch_size = hidden1_large.shape[0]

        labels_idx = (torch.arange(batch_size, dtype=torch.long) + batch_size  * torch.distributed.get_rank()).to(self.device) 
        labels = F.one_hot(labels_idx, enlarged_batch_size*2).to(self.device) 
        masks  = F.one_hot(labels_idx, enlarged_batch_size).to(self.device) 
        batch_size = enlarged_batch_size
    else:
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(self.device) 
        masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(self.device) 

    logits_aa = torch.matmul(hidden1, hidden1_large.T)
    logits_aa = logits_aa - masks * self.LARGE_NUM
    logits_bb = torch.matmul(hidden2, hidden2_large.T)
    logits_bb = logits_bb - masks * self.LARGE_NUM
    logits_ab = torch.matmul(hidden1, hidden2_large.T)
    logits_ba = torch.matmul(hidden2, hidden1_large.T)

    #  SogCLR
    neg_mask = 1-labels
    logits_ab_aa = torch.cat([logits_ab, logits_aa ], 1)
    logits_ba_bb = torch.cat([logits_ba, logits_bb ], 1)
    
    neg_logits1 = torch.exp(logits_ab_aa /self.T)*neg_mask   #(B, 2B)
    neg_logits2 = torch.exp(logits_ba_bb /self.T)*neg_mask

    # u init    
    if self.u[index.cpu()].sum() == 0:
        gamma = 1
        
    u1 = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * torch.sum(neg_logits1, dim=1, keepdim=True)/(2*(batch_size-1))
    u2 = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * torch.sum(neg_logits2, dim=1, keepdim=True)/(2*(batch_size-1))

    # this sync on all devices (since "hidden" are gathering from all devices)
    if distributed:
        u1_large = concat_all_gather(u1)
        u2_large = concat_all_gather(u2)
        index_large = concat_all_gather(index)
        self.u[index_large.cpu()] = (u1_large.detach().cpu() + u2_large.detach().cpu())/2 
    else:
        self.u[index.cpu()] = (u1.detach().cpu() + u2.detach().cpu())/2 

    p_neg_weights1 = (neg_logits1/u1).detach()
    p_neg_weights2 = (neg_logits2/u2).detach()

    def softmax_cross_entropy_with_logits(labels, logits, weights):
        expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/(2*(batch_size-1))
        normalized_logits = logits - expsum_neg_logits
        return -torch.sum(labels * normalized_logits, dim=1)

    loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1)
    loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2)
    loss = (loss_a + loss_b).mean()

    return loss