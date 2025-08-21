"""
Reference: https://github.com/Sara-Ahmed/SiT

# keypoints:
- reconstruction for patch tokens
- contrastive learning for CLS token
"""
from functools import partial

import timm.utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers.pos_embed import resample_abs_pos_embed
from util.pos_embed import get_2d_sincos_pos_embed
import gin
import timm
# from model.operation import build_mlp, contrastive_loss
# from model.models_mae import MaskedAutoencoderViT,build_mae_backbone




@gin.configurable()
class SiT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone    
    """
    def __init__(self, 
                 model_name = 'vit_small',
                 out_dim = 256,
                 mlp_dim=1024,
                 momentum=0.996,
                 lambd=1,
                 T = 0.2,
                 ):
        super().__init__()        
        self.lambd = lambd
        self.T = T
        self.student = build_mae_backbone(model_name)
        self.embed_dim = self.student.embed_dim
        self._teacher = timm.utils.ModelEmaV2(self.student,momentum)
        self._teacher.module.mask_ratio=0
        self._teacher.requires_grad_(False)
        # projector
        self.projector = build_mlp(2, self.embed_dim,mlp_dim,out_dim)
        teacher_projector = timm.utils.ModelEmaV2(self.projector,momentum)
        teacher_projector.requires_grad_(False)
        self.teacher_projector = teacher_projector
        # predictor
        self.predictor = build_mlp(2,out_dim,mlp_dim, out_dim,False)
        
    @torch.no_grad()
    def teacher(self,x):
        latent, mask, ids_restore =  self._teacher.module.forward_encoder(x,0)
        return self.teacher_projector.module(latent[:,0])
    
    def update(self):
        self._teacher.update(self.student)
        self.teacher_projector.update(self.projector)

    def group_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # TODO
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def representation(self, x, pos_embed=None):  
        if isinstance(x,list):
            x= x[0]      
        return self._teacher.module.representation(x,pos_embed)

    def forward(self, imgs, **kwargs):
        self.log = {}
        x1, x2 = imgs[:2]
        local_x = imgs[2:]

        recon1,z1 = self.student(x1,return_variables=True)
        recon2,z2 = self.student(x2,return_variables=True)
        q1 = self.predictor(self.projector(z1[:,0]))
        q2 = self.predictor(self.projector(z2[:,0]))
        k1 = self.teacher(x1)
        k2 = self.teacher(x2)
        
        loss_rec = (recon1+recon2)/2
        loss_cl = ( contrastive_loss(q1, k2, self.T) + 
                contrastive_loss(q2, k1, self.T))/2
        self.log.update({'loss_cl':loss_cl.item(),'loss_rec':loss_rec.item()})
        loss = self.lambd * loss_cl + loss_rec
        return loss,self.log
