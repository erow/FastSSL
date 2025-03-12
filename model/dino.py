"""
Reference: https://github.com/facebookresearch/dino

# Keypoints
- Center: the key to avoiding collapse.
- DINO head: WeightNorm is applied at the end and the weight_g (magnitude) is fixed. Therefore, it only optimizes the direction, which equals to L2-normalization. In addition, BN is removed. L2-normalization bottleneck stabilizes the training of DINO with deep projection head.
- freeze last layer: the .last_layer in 
- output dimension: large output dimensionality improves the performance. 65536 is the best.


# Result:


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import gin
from timm.layers import trunc_normal_

from layers.backbone import create_backbone

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
    def __init__(self, 
                 embed_dim = 2048,
                 norm_last_layer=True,
                 out_dim=60000, 
                 teacher_temp=0.05, student_temp=0.1,
                 center_momentum=0.9):
        """
        dim: feature dimension (default: 60000)
        teacher_temp: softmax temperature for teacher. Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend  starting with the default value of 0.04 and increase this slightly if needed.
        student_temp: 
        """
        super().__init__()
        self.out_dim = out_dim
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp # TODO: adjust the temperature dynamically for the teacher

        # build encoders
        backbone = create_backbone()        
        projector = DINOHead(embed_dim, out_dim, norm_last_layer=norm_last_layer)
        
        self.embed_dim = embed_dim
        self.student = nn.Sequential(backbone,projector)
        
        _teacher = nn.Sequential(create_backbone(),
                                 DINOHead(embed_dim, out_dim,))
        _teacher.requires_grad_(False)
        self._teacher = _teacher
        self.update(0)

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
    def update(self,m):
        for ema_v, model_v in zip(self._teacher.state_dict().values(), self.student.state_dict().values()):
            ema_v.data.mul_(m).add_((1 - m) * model_v.detach().data)

    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        latent = self._teacher[0](x)
        proj = self._teacher[1](latent)
        return dict(latent=latent,proj=proj)

    def forward(self, imgs, **kwargs):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        self.log = {}
        x1, x2 = imgs[:2]
        local_x = imgs[2:]

        # predict the distribution of clusters
        logit1 = self.student(x1)
        logit2 = self.student(x2)
        
        with torch.no_grad():
            t_output1 = self.teacher(x1)
            t_output2 = self.teacher(x2)
            # teacher centering and sharpening
            q1 = F.softmax((t_output1 - self.center)/self.teacher_temp,dim=-1)
            q2 = F.softmax((t_output2 - self.center)/self.teacher_temp,dim=-1)
            t_output = (t_output1 + t_output2)/2
            self.update_center(t_output)         

        loss = (
            torch.sum(- q1 * F.log_softmax(logit2/self.student_temp,dim=-1),-1) + 
            torch.sum(- q2 * F.log_softmax(logit1/self.student_temp,dim=-1),-1)
        ).mean()/2

        loss_local = 0
        for lx in local_x:
            lz = self.student(lx)

            loss_local += (
                torch.sum(- q1 * F.log_softmax(lz/self.student_temp),-1) + 
                torch.sum(- q2 * F.log_softmax(lz/self.student_temp),-1)
            ).mean()/2


        p1 = F.softmax(logit1/self.student_temp,dim=-1)
        p2 = F.softmax(logit2/self.student_temp,dim=-1)
        K = (p2.shape[1])
        
        # CE = q(z) * log(q(z|x)) = H(z) - KL(q(z)||q(z|x)), KL(q(z)||q(z|x)) > H(q(z|x))
        # minimizing CE causes collapse H(Z) -> 0 and reduce the MI(Z,X)
        self.log['z_ce'] = - (p1 * torch.log(p2)).sum(-1).mean().item()
        self.log['H_zcx'] = - (p1 * torch.log(p1)).sum(-1).mean().item()
        # self.log['MI'] =  - (q1 * torch.log(p1)).sum(-1).mean().item()
        
        return loss,self.log
    