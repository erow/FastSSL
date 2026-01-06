"""
torchrun main_pretrain.py --data_set cifar10 --data_path ../data/  --batch_size 512 --epochs=1000 --warmup_epochs=10 --ckpt_freq 20 --lr=1e-4  --min_lr=1e-4  --cfgs configs/cifar.gin configs/resnet18.gin --gin build_model.embed_dim=512 build_model.model_fn=@SimLAP

torchrun main_pretrain.py --data_set ffcv --data_path $train_path --batch_size 512 --epochs=1000 --warmup_epochs=10 --ckpt_freq 20 --lr=1e-4  --min_lr=1e-4  --cfgs configs/simclr_ffcv.gin --gin build_model.embed_dim=2048 build_model.model_fn=@SimLAP
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
import gin
import numpy as np
from layers.operation import *
from layers.backbone import create_backbone

def multipos_ce_loss(logits, pos_mask,exclude_mask=None):
    if exclude_mask is None:
        exclude_mask = pos_mask
    logits = logits - logits.mean(1,keepdim=True).detach()
    score = logits.exp()
    N = score.size(0)
 
    # InfoNCE loss 
    ## exclude the positives and class pairs
    neg = (score*(~exclude_mask)).sum(1,keepdim=True)
    loss = torch.sum(pos_mask* (torch.log(score + neg) - logits))/pos_mask.sum()
    loss = loss.mean()
   
    return loss

def apply_gate(gate, x1, x2):
    x1 = torch.einsum("bk,bk->bk",x1,gate)
    x2 = torch.einsum("nk,bk->bnk",x2,gate)
    x1 =  F.normalize(x1,p=2,dim=-1)
    x2 =  F.normalize(x2,p=2,dim=-1)
    return x1, x2

def contrast(x1,x2):
    if x2.dim() == 3:
        logits =  torch.einsum("bj,bnj->bn",x1,x2)
    else:
        logits = x1 @ x2.t()
    return logits

@gin.configurable()
class OpenGate(nn.Module):
    def __init__(self, embed_dim,num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def forward(self,y1,y2=None,log=None):
        bs = y1.size(0)
        gate = torch.ones(bs,self.embed_dim,device=y1.device)
        return gate

@gin.configurable()
class BasicGate(OpenGate):
    def __init__(self, embed_dim, num_classes=1000, in_dim=512, mlp_dim=1024,
                 lam = 0, fuse=True):
        super().__init__(embed_dim,num_classes)
        
        self.mlp = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(), nn.BatchNorm1d(mlp_dim),
            nn.Linear(mlp_dim, embed_dim),            
        )
        self.label_embedding = nn.Embedding(num_classes,in_dim)
        self.lam = lam
        self.fuse = fuse

    def statistics(self):
        labels = torch.arange(self.num_classes).cuda()
        label_embeds = self.label_embedding(labels)
        logits = self.mlp(label_embeds)
        gates = logits.sigmoid()
        activation = gates.sum(1).mean()
        entropy = torch.distributions.Bernoulli(gates).entropy().mean()
        return activation, entropy
    
    def forward(self,y1,y2=None,log=None):
        if self.fuse:      
            if y2 is None:
                label_embeds = self.label_embedding(y1)
            else:
                label_embeds = (self.label_embedding(y1) + self.label_embedding(y2))/2
            
            logits = self.mlp(label_embeds)
            gate = logits.sigmoid()
        else:
            if y2 is None:
                gate = self.mlp(self.label_embedding(y1)).sigmoid()
            else:
                gate1 = self.mlp(self.label_embedding(y1)).sigmoid()
                gate2 = self.mlp(self.label_embedding(y2)).sigmoid()
                gate = gate1 * gate2
        return gate


def reparameterize_with_gumbel_softmax(logits, tau=1.0, hard=False):
    """
    Apply the Gumbel-Softmax trick to reparameterize the Bernoulli distribution.
    logits: Logits from which to sample (e.g., the output of a neural network).
    tau: Temperature parameter. Lower values make samples more discrete.
    hard: If True, use straight-through Gumbel-Softmax Estimator.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

    if hard:
        y_hard = torch.round(y_soft)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y

@gin.configurable
class GumbelGates(BasicGate):
    def forward(self, y1,y2=None,log=None):
        if y2 is None:
            label_embeds = self.label_embedding(y1)
        else:
            label_embeds = (self.label_embedding(y1) + self.label_embedding(y2))/2
        logits = self.mlp(label_embeds)
        gate = reparameterize_with_gumbel_softmax(logits)

        if not log is None:
            p = logits.sigmoid().detach().clamp(1e-6,1-1e-6)
            dist = torch.distributions.Bernoulli(p)
            entropy = dist.entropy()
            log['entropy'] = entropy.mean().item()
            open = (gate>0.5).float()
            log['activation']=(open.sum(0)>20).float().sum().item()
            
        return gate
    
@gin.configurable()
class StochasticGate(BasicGate):
    """Paper: Feature Selection using Stochastic Gates
    Code: https://github.com/runopti/stg/blob/master/python/stg/models.py
    """
    def __init__(self, embed_dim, num_classes=1000, in_dim=512, mlp_dim=1024,
                sigma=0.5,lam=1e-6):
        super().__init__(embed_dim, num_classes,in_dim,mlp_dim)
        self.sigma=sigma
        self.lam = lam
        
    def forward(self,y1,y2=None,log=None):
        if y2 is None:
            label_embeds = self.label_embedding(y1)
        else:
            label_embeds = (self.label_embedding(y1) + self.label_embedding(y2))/2
        mu = self.mlp(label_embeds)
        noise = torch.randn_like(mu)*self.sigma
        gate = self.hard_sigmoid(mu+noise)
        if not log is None:
            reg = self.regularizer((mu+0.5)/self.sigma).sum(1).mean()
            log['reg'] = reg
            log['mu']=mu.mean().item()+0.5
            # log['lam']=self.lam
        return gate


    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 
    
    def reg(self,mu):
        reg = torch.mean(self.regularizer((mu + 0.5)/self.sigma)) 
        return reg


@gin.configurable()
class Filter(nn.Module):
    def __init__(self,num_classes, embed_dim, gate_fn=BasicGate):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate = gate_fn(embed_dim,num_classes=num_classes)
    
    def forward(self, x1,x2,y1,y2=None):
        gate = self.gate(y1,y2)
        x1 = torch.einsum("bk,bk->bk",x1,gate)
        x2 = torch.einsum("nk,bk->bnk",x2,gate)
        x1 =  F.normalize(x1,p=2,dim=-1)
        x2 =  F.normalize(x2,p=2,dim=-1)
        return x1, x2
    
    def contrast(self,x1,x2):
        logits =  torch.einsum("bj,bnj->bn",x1,x2)
        return logits

@gin.configurable
class SimLAP(nn.Module):
    def __init__(self,
                 out_dim=256,
                 embed_dim=2048,
                 mlp_dim=2048, 
                 type='arbitrary',
                 temperature=0.1,
                 num_classes=1000,
                 alpha = 0,
                 ):
        super(SimLAP, self).__init__()
        self.num_classes = num_classes
        assert type in ['arbitrary','identical','distinct']
        self.scale_logit = nn.Parameter(torch.zeros(1)+np.log(1/temperature))
        self.out_dim = out_dim
        self.type = type
        if alpha > 0:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.alpha = None
        self.embed_dim = embed_dim
        self.backbone = create_backbone()
        self.projector = build_head(2,embed_dim,mlp_dim,out_dim, last_norm='ln')
        # self.filter = Filter(num_classes=num_classes,embed_dim=out_dim)
        self.filter = BasicGate(out_dim, num_classes=num_classes)

    @torch.no_grad()
    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        latent = self.backbone(x)
        proj = self.projector(latent)
        rep = dict(latent=latent,proj=proj)
        return rep

    def forward(self, samples, targets, **kwargs):
        self.log = {}
        x1,x2 = samples[:2]
        
        local_x = samples[2:]

        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        
        y1 = targets
        if self.type == 'identical':
            y2 = targets
        elif self.type == 'distinct':
            y2 = (targets + torch.randint(1,self.num_classes,(len(targets),),device=targets.device))%self.num_classes
        elif self.type == 'arbitrary':
            y2 = targets[torch.randperm(len(targets),device=targets.device)]
        else:
            raise ValueError(f"Invalid type: {self.type}")

        loss = self.disparate_loss(z1,z2,y1,y2)

        self.log['z@sim'] = F.cosine_similarity(z1,z2).mean().item()

        return loss, self.log
    
   
    def disparate_loss(self, z1, k2, y1, posy):
        k2 = concat_all_gather_grad(k2)
        gate = self.filter(y1,posy)
        z1 = F.normalize(z1,p=2,dim=-1)
        k2 = F.normalize(k2,p=2,dim=-1)
        scale = self.scale_logit.exp()
        logits = contrast(z1*gate*gate,k2) * scale 
        # fz1,fz2 = apply_gate(gate, z1, k2)
        # logits = contrast(fz1,fz2) * self.s
        if self.alpha is not None:
            logits = logits - self.alpha * contrast(z1,k2)

        all_y1 = concat_all_gather(y1)
        c1_mask = (y1.unsqueeze(1) == all_y1.unsqueeze(0)) # exclude samples from y1
        c2_mask = (posy.unsqueeze(1) == all_y1.unsqueeze(0)) # exclude samples from y2
        class_mask = c1_mask|c2_mask

        loss = multipos_ce_loss(logits, c2_mask, class_mask)
        
        self.log['activation'] = gate.sum(1).mean().item()
        entropy = torch.distributions.Bernoulli(gate).entropy().mean()
        self.log['entropy'] = entropy.item()
        self.log['scale'] = scale.item()
        return loss
