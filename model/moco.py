"""
Reference: https://github.com/facebookresearch/moco-v3

# Note
MoCo v3 consists of backbone, projector, predictor, and their momentum version, where projector is updated by EMA, predictor is learnable. There are following key points

- Normalization Layer at the end of Projector: 2048-4096-4096-256-BN or LN
- Predictor: a MLP ending without BN. 256-4096-256
- parameter group?


## [train](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md)

ResNet50
```python
torchrun --nproc_per_node=8 main_pretrain_ema.py --batch_size=128 --opt LARS --blr=5e-2 --weight_decay=1.5e-6 --epochs=100 --warmup_epochs=10 --ckpt_freq=100 --data_path $FFCVTRAIN --data_set ffcv --gin  build_dataset.transform_fn=@MultiviewPipeline MultiviewPipeline.scale="(0.2,1)" build_model.model_fn=@MoCo MoCo.T=1 -m 0.99 
```

ViT-Base, 1-node (8-GPU) pre-training (bs=4K).
```
torchrun --nproc_per_node=8 main_pretrain_ema.py --batch_size=512 --opt adamw --blr=1.5e-4 --opt_betas 0.9 0.999 --weight_decay=.1 \
    --epochs=300 --warmup_epochs=40 --ckpt_freq=20 --data_path $IMNET \
    -m 0.99 --gin build_dataset.transform_fn=@DataAugmentationDINO DataAugmentationDINO.local_crops_number=0 build_model.model_fn=@MoCo MoCo.T=0.2 MoCo.stop_grad=True MoCo.embed_dim=768 create_backbone.name=\"vit_base_patch16_224\" 
```

# Official Result:
| Model    | pretrain epochs | pretrain crops | linear acc |   |
|----------|:---------------:|:--------------:|:----------:|:-:|
| resnet50 | 100             | 2x224          | 68.9       |   |
| resnet50 | 300             | 2x224          | 72.8       |   |
| resnet50 | 1000            | 2x224          | 74.6       |   |

# our impl. 
CIFAR10(k=10),vit_base
- orig: 95
- FFCV, bs=4k: 74.96
- IF, bs=2K: 91.68
- IF, weight init,bs=4k: 92.4
- optim: 

"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin

from layers.backbone import create_backbone
from layers.operation import build_head, contrastive_loss

import math
from functools import partial, reduce
from operator import mul


def mocov3_init(model, stop_grad_conv1=True, ):
    # Use fixed 2D sin-cos position embedding
    build_2d_sincos_position_embedding(model)

    # weight initialization
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'qkv' in name:
                # treat the weights of Q, K, V separately
                val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                nn.init.uniform_(m.weight, -val, val)
            else:
                nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    nn.init.normal_(model.cls_token, std=1e-6)

    # xavier_uniform initialization
    val = math.sqrt(6. / float(3 * reduce(mul, model.patch_embed.patch_size, 1) + model.embed_dim))
    nn.init.uniform_(model.patch_embed.proj.weight, -val, val)
    nn.init.zeros_(model.patch_embed.proj.bias)

    if stop_grad_conv1:
        model.patch_embed.proj.weight.requires_grad = False
        model.patch_embed.proj.bias.requires_grad = False

def build_2d_sincos_position_embedding(model, temperature=10000.):
    h, w = model.patch_embed.grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert model.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = model.embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    model.pos_embed[:,1:].data = pos_emb
    model.pos_embed.requires_grad = False



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
                 stop_grad=True,
                 T=0.2,):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 0.2)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.m = 0

        # build encoders
        backbone = create_backbone()
        mocov3_init(backbone, stop_grad_conv1=stop_grad)
        if stop_grad and hasattr(backbone, 'patch_embed'):
            backbone.patch_embed.requires_grad_(False)            
        
        self.embed_dim = embed_dim
        projector = build_head(3,embed_dim,mlp_dim,out_dim)
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

if __name__ == '__main__':
    x = torch.randn(2,3,224,224)
    gin.parse_config([
        "create_backbone.name='vit_base_patch16_224'",
    ])
    model = MoCo(embed_dim=768, out_dim=256, mlp_dim=4096, stop_grad=True)
    model([x,x])