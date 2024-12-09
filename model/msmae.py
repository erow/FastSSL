from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from timm.layers.pos_embed import resample_abs_pos_embed
from util.pos_embed import get_2d_sincos_pos_embed
import gin
from .target import build_target
from model.models_mae import MaskedAutoencoderViT

@gin.configurable()
class MultiScaleMAE(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone    
    """
    def forward_decoder(self, x,pos_embed=None):
        if pos_embed is None:
            pos_embed = self.decoder_pos_embed
        # embed tokens
        x = self.decoder_embed(x)
        # exclude cls token
        pos_embed = pos_embed[:,1:]
        # append mask tokens to sequence
        x_ = self.mask_token + pos_embed.expand(x.shape[0], -1, -1)
        x = torch.cat([x, x_], dim=1)  # append cls token
        len_pred = x_.shape[1]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, -len_pred:, :]

        return x
    
    def forward(self, imgs, return_variables=False,**kwargs):
        if isinstance(imgs, list) or isinstance(imgs, tuple):
            imgs = imgs[0]
        ## dynamic pos embed
        self.log = {}
        B, C, H, W = imgs.shape
        lx = imgs
        sx = F.interpolate(lx, scale_factor=0.5, mode='bilinear', align_corners=False)

        patch_size = self.patch_embed.patch_size[0]

        lpos_embed = resample_abs_pos_embed(
            self.pos_embed,
            (H//patch_size, W//patch_size),
        )
        ldecoder_pos_embed = resample_abs_pos_embed(
            self.decoder_pos_embed,
            (H//patch_size, W//patch_size),
        )

        spos_embed = resample_abs_pos_embed(
            self.pos_embed,
            (H//2//patch_size, W//2//patch_size),
        )
        sdecoder_pos_embed = resample_abs_pos_embed(
            self.decoder_pos_embed,
            (H//2//patch_size, W//2//patch_size),
        )
        # pos_embed, decoder_pos_embed = self.pos_embed, self.decoder_embed
        ## dynamic pos embed
        
        mask_ratio= self.mask_ratio

        llatent, lmask, lids_restore = self.forward_encoder(lx, mask_ratio,lpos_embed)
        slatent, smask, sids_restore = self.forward_encoder(sx, 0, spos_embed)

        # cross prediction
        spred = self.forward_decoder(llatent, sdecoder_pos_embed) 
        lpred = self.forward_decoder(slatent, ldecoder_pos_embed)
        
        sloss = self.target_loss(sx, spred)
        lloss = self.target_loss(lx, lpred)

        loss = (sloss + lloss)/2
        self.log.update({
            'rec': sloss.item(),
            'sr': lloss.item(),            
        })
        if return_variables:
            return loss, llatent, slatent, lmask, smask
        else:
            return loss,self.log
        
@gin.configurable()
def msmae_tiny_patch16(**kwargs):
    model = MultiScaleMAE(
        patch_size=16,embed_dim=196,depth=12,num_heads=12,
        decoder_embed_dim=96,decoder_depth=1,decoder_num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@gin.configurable()
def msmae_small_patch16(**kwargs):
    model = MultiScaleMAE(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, 
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@gin.configurable()
def msmae_base_patch16(**kwargs):
    model = MultiScaleMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@gin.configurable()
def msmae_large_patch16(**kwargs):
    model = MultiScaleMAE(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@gin.configurable()
def msmae_huge_patch14(**kwargs):
    model = MultiScaleMAE(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
