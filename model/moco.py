"""
Reference: https://github.com/facebookresearch/moco-v3

# Note
- Projector: a MLP ending with BN. 2048-4096-256
- Predictor: a MLP ending without BN. 256-4096-256

## [train](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md)

ResNet50
`torchrun --nproc_per_node=8 main_pretrain_ema.py --batch_size=128 --opt LARS --blr=5e-2 --weight_decay=1.5e-6 --epochs=100 --warmup_epochs=10 --ckpt_freq=100 --data_path $train_path --prob_path $val_path --gin  build_dataset.transform_fn=@MultiviewPipeline MultiviewPipeline.scale="(0.2,1)" build_model.model_fn=@MoCo MoCo.T=1 -m 0.99 `


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
|          |                 |                |            |   |
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin

from layers.backbone import create_backbone
from layers.operation import build_head, contrastive_loss

@gin.configurable
class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, 
                 embed_dim = 2048,
                 out_dim=256, 
                 mlp_dim=4096, 
                 T=1.0,):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.m = 0

        # build encoders
        backbone = create_backbone()
        
        self.embed_dim = embed_dim
        projector = build_head(2,embed_dim,mlp_dim,out_dim)
        self.student = nn.Sequential(backbone,                                     
                                     projector)

        self._teacher = deepcopy(self.student)
        self._teacher.requires_grad_(False)
        self.update(0)

        self.predictor = build_head(2,out_dim,mlp_dim,out_dim,False)

    @torch.no_grad()
    def teacher(self,x):
        return self._teacher(x)
    
    @torch.no_grad()
    def update(self,m):
        for param_b, param_m in zip(self.student.parameters(), self._teacher.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    @torch.no_grad()
    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        latent = self._teacher[0](x)
        proj = self._teacher[1](latent)
        s_latent = self.student[0](x)
        s_proj = self.student[1](s_latent)
        rep = dict(latent=latent,proj=proj,s_latent=s_latent,s_proj=s_proj)
        return rep

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
        self.log = {}

        # compute features
        z1 = self.student(x1)
        q1 = self.predictor(z1)
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

        self.log['qk@sim'] = F.cosine_similarity(q1,k1).mean().item()
        self.log['z@sim'] = F.cosine_similarity(q1,k2).mean().item()
        return loss,self.log


class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
        x = self.norm(x)
        return x