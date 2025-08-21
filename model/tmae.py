# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from timm.layers.pos_embed import resample_abs_pos_embed
from layers.operation import patchify
from util.pos_embed import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
import gin
from torchvision.transforms import GaussianBlur
from layers.target import build_target
import torchvision.transforms.functional as tfF
# recipe https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md

def apply_ids(x, index,dim=1):
    """
    Apply ids to x
    x: [N, L, D], sequence
    ids: [N, L], index
    """
    N, L, D = x.shape  # batch, length, dim
    index = index.unsqueeze(-1).repeat(1, 1, D)
    x = torch.gather(x, dim=dim, index=index)
    return x

def random_masking_idx(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    # ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_pred = ids_shuffle[:, len_keep:]
    x_masked = apply_ids(x, index=ids_keep)

    return x_masked, ids_keep, ids_pred

class TemporalMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone    
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_frames=2, augmentation=None, drop_ratio=0,
                 mask_ratio=0.75,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, decoder_feature_size=None,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.augmentation = augmentation
        self.drop_ratio = drop_ratio
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.img_size=img_size
        self.num_frames = num_frames
        self.norm_pix_loss = True
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, strict_img_size=False)
        num_patches = (img_size // patch_size) * (img_size // patch_size) 

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        # add temporal information
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames,1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, torch.arange(num_frames), )
        self.temporal_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().reshape_as(self.temporal_pos_embed))
        
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_pred = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio:float,):
        B, C, H, W = x.shape
                
        pos_embed = self.pos_embed
        
        # cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_token = cls_token.expand(B, -1, -1)                
        
        # patches tokens
        t0 = self.patch_embed(x)

        # add pos embed w/o cls token
        t0 = t0 + self.pos_embed[:, 1:, :]
        
        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(t0, mask_ratio)
        x, ids_mask,ids_pred = random_masking_idx(t0, mask_ratio)
        
        # add temporal information
        t0 = x + self.temporal_pos_embed[:, 0]  # [N, L, D]        
        t1 = x + self.temporal_pos_embed[:, 1]
        
        # add cls token
        x = torch.cat((cls_token, t0, t1), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_mask,ids_pred

    def forward_decoder(self, x, ids_mask,ids_pred, pos_embed=None):
        if pos_embed is None:
            pos_embed = self.decoder_pos_embed
        B = len(x)
        
        # embed tokens
        N,L = ids_mask.shape
        x = self.decoder_embed(x)
        x_cls,t0,t1 = x.split([1, L,L], dim=1)                      
        
        # pos embed
        pos_embed1 = pos_embed[:,1:].repeat(N,1,1)
        pred_pos_embed = apply_ids(pos_embed1, ids_pred, dim=1)  # [N, L, D]
        mask_pos_embed = apply_ids(pos_embed1, ids_mask, dim=1)  # [N, L, D]

        # add pos embed
        x_cls = x_cls + pos_embed[:, :1, :]  # [N, 1, D]
        t0 = t0 + mask_pos_embed # [N, L, D]
        t1 = t1 + mask_pos_embed
        pred_tokens = self.mask_token + pred_pos_embed 
                
        x = torch.cat([x_cls, t0, t1, pred_tokens], dim=1)  # append cls token
        

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token and masked tokens
        x = x[:, 1+L+L:, :]

        return x



    def forward(self, imgs,**kwargs):
        self.log = {}
        if isinstance(imgs, list) or isinstance(imgs, tuple):
            imgs = imgs[0]
        ## dynamic pos embed
        B, C, H, W = imgs.shape

        ## dynamic pos embed
        
        mask_ratio= self.mask_ratio

        # latent, mask, ids_restore = self.forward_encoder(imgs,imgs1, mask_ratio)
        latent, ids_mask,ids_pred = self.forward_encoder(imgs, mask_ratio)
        
        
        pred = self.forward_decoder(latent, ids_mask, ids_pred)  # [N, L + L + S, p*p*3]        
                        
        target = patchify(imgs,self.patch_embed.patch_size[0])
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        pred_target = apply_ids(target, ids_pred, dim=1)  # [N, S, D]
        
        loss = (pred - pred_target) ** 2
        loss = loss.mean()  # [N, T, L], mean loss per patch
            
        return loss, self.log

@gin.configurable()
def tmae_tiny(**kwargs):
    default_cfg = dict(
        patch_size=16,embed_dim=192,depth=12,num_heads=12,
        decoder_embed_dim=96,decoder_depth=1,decoder_num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    default_cfg.update(kwargs)
        
    model = TemporalMaskedAutoencoderViT(**default_cfg)
    return model

@gin.configurable()
def tmae_small(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, 
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = TemporalMaskedAutoencoderViT(**default_cfg)
    return model

@gin.configurable()
def tmae_base(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = TemporalMaskedAutoencoderViT(**default_cfg)
    return model

@gin.configurable()
def tmae_large(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = TemporalMaskedAutoencoderViT(**default_cfg)
    return model

@gin.configurable()
def tmae_huge_patch14_dec512d8b(**kwargs):
    default_cfg = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = TemporalMaskedAutoencoderViT(**default_cfg)
    return model

if __name__=="__main__":
    x = torch.rand(10,3,224,224)
    model = tmae_base()
    model(x)