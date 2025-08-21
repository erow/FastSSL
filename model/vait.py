"""
Reference: https://github.com/facebookresearch/vait

# Note
- no weight decay for bias
- adamw betas = 0.9 0.95

# Training
```python

--opt_filter
```

# Result

"""


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from timm.layers.pos_embed import resample_abs_pos_embed
from util.pos_embed import get_2d_sincos_pos_embed
import gin
from torchvision.transforms import GaussianBlur
from layers.target import build_target
# recipe https://github.com/facebookresearch/vait/blob/main/PRETRAIN.md

class VariationalVisionTrasformer(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone    
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 mask_ratio: float=0.75, 
                 beta=1e-3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, decoder_feature_size=None,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.beta = beta
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.img_size=img_size
        # --------------------------------------------------------------------------
        # vait encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, strict_img_size=True)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.vae_proj = nn.Linear(embed_dim, embed_dim*2, bias=True)  # VAE projection to mu and logvar
        # --------------------------------------------------------------------------
        # vait decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if decoder_feature_size is None:
            decoder_feature_size = patch_size**2 * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_feature_size, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.target_loss = build_target(patch_size=patch_size)
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    
    def representation(self, x, pos_embed=None):
        B, C, H, W = x.shape
        ## dynamic pos embed
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            (H//self.patch_embed.patch_size[0], W//self.patch_embed.patch_size[1]),
        )
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]


        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # x = self.norm(x) remove normalization

        return x[:,1:].mean(1)

    def forward_encoder(self, x, pos_embed=None):
        B, C, H, W = x.shape
        if pos_embed is None:
            pos_embed = self.pos_embed
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]
        
        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    
    
    def forward_decoder(self, x, pos_embed=None):
        if pos_embed is None:
            pos_embed = self.decoder_pos_embed        
        x = self.decoder_embed(x)  # [N, L, decoder_embed_dim]
        # add pos embed
        x = x + pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x



    def forward(self, imgs,**kwargs):
        self.log = {}
        if isinstance(imgs, list) or isinstance(imgs, tuple):
            imgs = imgs[0]
                   
        x = self.forward_encoder(imgs)   
        mu,logvar = self.vae_proj(x).chunk(2,dim=2)
        z = mu + torch.rand_like(logvar)*(logvar/2).exp()
        kl_loss = compute_gaussian_kl(mu,logvar,axis=None).mean()
        self.log['kl_loss'] = kl_loss
        
        # mask features
        m = self.mask_ratio
        mask = torch.rand_like(z)<m
        e = torch.randn_like(z)
        c = torch.where(mask, e, z)
        c = c / (1-m)  # this makes sure that the similarity of z*c is the same as changing m.        
        
        pred = self.forward_decoder(c)  # [N, L, p*p*3]     
        target = self.patchify(imgs)
           
        recon_loss = F.mse_loss(pred,target)
        self.log['recon_loss'] = recon_loss
        
        loss = recon_loss + self.beta * kl_loss
        return loss, self.log


def compute_gaussian_kl(z_mean, z_logvar,axis=0):
    """Compute KL divergence between input Gaussian and Standard Normal."""
    kl =torch.square(z_mean) + torch.exp(z_logvar) - z_logvar - 1
    if axis is None:
        return 0.5 * kl
    else:
        return 0.5 * torch.mean(kl, axis)
    
@gin.configurable()
def vait_tiny(**kwargs):
    default_cfg = dict(
        patch_size=16,embed_dim=192,depth=12,num_heads=3,
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    default_cfg.update(kwargs)
        
    model = VariationalVisionTrasformer(**default_cfg)
    return model

@gin.configurable()
def vait_small(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, 
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = VariationalVisionTrasformer(**default_cfg)
    return model

@gin.configurable()
def vait_base(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = VariationalVisionTrasformer(**default_cfg)
    return model

@gin.configurable()
def vait_large(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = VariationalVisionTrasformer(**default_cfg)
    return model

@gin.configurable()
def vait_huge_patch14_dec512d8b(**kwargs):
    default_cfg = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = VariationalVisionTrasformer(**default_cfg)
    return model

if __name__ == '__main__':
    model = vait_small()
    