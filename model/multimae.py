"""


torchrun --nproc_per_node=8 main_pretrain.py --data_path=$FFCVTRAIN --data_set=ffcv --epochs 800 --warmup_epochs 40 --opt adamw --opt_betas 0.9 0.95 --blr 1.5e-4 --weight_decay 0.05 --batch_size 512 --gin build_model.model_fn=@rmae_base build_dataset.transform_fn=@SimplePipeline  --ckpt_freq=100 
"""

from functools import partial

import numpy as np
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

def interpolate_pos_idx(indices, width,scale=2):
    """
    Interpolate indices to a new scale.
    args:
        indices: [N, L], indices of the patches
        width: int, width of the patches
    return:
        new_indices: [N, L*scale*scale], interpolated indices
    """
    # TODO: fix scale ratio
    i = indices //width
    j = indices % width
    
    new_indices= [
        4*i*width + 2*j,
        4*i*width + 2*j+1,
        4*i*width + 2*width + 2*j,
        4*i*width + 2*width + 2*j+1
    ]
    new_indices = torch.cat(new_indices,dim=1)
    return new_indices
    
def random_masking_idx(N,  token_accum, mask_ratio,device='cuda'):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    """
    L = 49 # TODO: fix size
    
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    # ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_pred = ids_shuffle[:, len_keep:]
    
    # for the second branch, we need to interpolate the indices
    ids_pred_interpolated = interpolate_pos_idx(ids_pred, int(np.sqrt(L)), scale=2)
    ids_keep_interpolated = interpolate_pos_idx(ids_keep, int(np.sqrt(L)), scale=2)
    ## second masking
    noise = torch.rand(N, ids_keep_interpolated.shape[1], device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)[:, :L//2]  # ascend: small is keep, large is remove
    ids_keep_interpolated = torch.gather(ids_keep_interpolated, index=ids_shuffle,dim=1)
    
    return [ids_keep,ids_keep_interpolated], [ids_pred,ids_pred_interpolated]

class MultiscaleMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone    
    """
    def __init__(self, img_sizes=[112,224], patch_size=16, in_chans=3,                 
                 mask_ratio=0.75, shared_embed=True,                 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, decoder_feature_size=None,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        img_size = img_sizes[-1]
        num_frames = len(img_sizes)
        self.img_sizes = img_sizes
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.img_size=img_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.norm_pix_loss = True
        self.shared_embed = shared_embed
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if shared_embed:
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, strict_img_size=False)
        else:
            self.patch_embeds = nn.ModuleList([
                PatchEmbed(s, patch_size, in_chans, embed_dim, strict_img_size=False) 
                for s in img_sizes])
        
        # calculate number of patches
        token_list = [1] + [int(s//16)**2 for s in img_sizes]
        token_accum = [np.sum(token_list[:i+1]) for i in range(len(token_list))]
        self.token_accum = token_accum
        num_patches = token_list[-1]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
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

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        decoder_layer = nn.TransformerDecoderLayer(decoder_embed_dim, decoder_num_heads, dim_feedforward=decoder_embed_dim*mlp_ratio, dropout=0.1, batch_first=True,norm_first=True)
        
        self.decoder_blocks = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if decoder_feature_size is None:
            decoder_feature_size = patch_size**2 * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_feature_size, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
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

    def forward_encoder(self, imgs, mask_ratio:float,):
        B, C, H, W = imgs.shape
        ids_mask_list,ids_pred_list = random_masking_idx(B, self.token_accum, mask_ratio)
        pos_embed = self.pos_embed
        
        # cls token
        cls_token = self.cls_token.expand(B, -1, -1)                
        tokens = [cls_token]
        
        for i,s in enumerate(self.img_sizes):
            ## dynamic pos embed
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (int(s)//self.patch_size, int(s)//self.patch_size),
                num_prefix_tokens=0,
            )
            # embed patches
            x = F.interpolate(imgs,size=s, mode='bilinear', align_corners=False)
            if self.shared_embed:
                x = self.patch_embed(x)
            else:
                x = self.patch_embeds[i](x) 
            x = x + pos_embed + self.temporal_pos_embed[:,i,:,:] 
            
            # masking: length -> length * mask_ratio      
            x = apply_ids(x, ids_mask_list[i], dim=1)  # [N, L, D]      
            tokens.append(x)
        # concat tokens
        x = torch.cat(tokens, dim=1)  # [B, L, D]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_mask_list, ids_pred_list

    def forward_decoder(self, tokens, mask_list,pred_list, pos_embed=None):
        if pos_embed is None:
            pos_embed = self.decoder_pos_embed
        
        visible_tokens = []
        accum = 0
        for i in range(self.num_frames):
            # embed tokens
            ids_mask = mask_list[i]  # [N, L]
            B,L = ids_mask.shape  # length of mask
            x = tokens[:, accum:accum + L, :]  # [N, L, D]
            x = self.decoder_embed(x)            
            accum += L
            
            s = self.img_sizes[i]
            ## dynamic pos embed
            pos_embed1 = resample_abs_pos_embed(
                self.decoder_pos_embed,
                (int(s)//self.patch_size, int(s)//self.patch_size),
                num_prefix_tokens=0,
            ).repeat(B, 1, 1)  # [N, L, D]
            # print(">>>>>>>>>>>>>>>>", pos_embed1.shape, ids_mask.shape, ids_pred.shape)
            vis_pos_embed = apply_ids(pos_embed1, ids_mask, dim=1)  # [N, L, D]
            # add pos embed
            x = x + vis_pos_embed # [N, L, D]
            
            visible_tokens.append(x)
        
        x = torch.cat(visible_tokens, dim=1) 
        pred_pos_embed = apply_ids(self.decoder_pos_embed.repeat(B,1,1), pred_list[-1], dim=1)  # [N, S-L, D]
        pred_tokens = self.mask_token + pred_pos_embed
        
        x = torch.cat([x, pred_tokens], dim=1)  # [N, S, D]
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove prefix
        x = x[:,accum:]
        return x



    def forward(self, imgs,**kwargs):
        self.log = {}
        if isinstance(imgs, list) or isinstance(imgs, tuple):
            imgs = imgs[0]
        
        mask_ratio= self.mask_ratio

        # latent, mask, ids_restore = self.forward_encoder(imgs,imgs1, mask_ratio)
        latent, mask_list,pred_list = self.forward_encoder(imgs, mask_ratio)        
        preds = self.forward_decoder(latent, mask_list, pred_list)  # [N, S+L, p*p*3]        
        
        loss = 0
        
        target = patchify(imgs,self.patch_embed.patch_size[0])
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        pred_target = apply_ids(target, pred_list[-1], dim=1)  # [N, S, D]
    
        
        mse = F.l1_loss(preds,  pred_target)  # [N, S, D], mean loss per patch
        loss += mse.mean()  # [N, T, L], mean loss per patch
        
        return loss, self.log

@gin.configurable()
def rmae_tiny(**kwargs):
    default_cfg = dict(
        patch_size=16,embed_dim=192,depth=12,num_heads=3,
        decoder_embed_dim=512,decoder_depth=4,decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    default_cfg.update(kwargs)
        
    model = MultiscaleMaskedAutoencoderViT(**default_cfg)
    return model

@gin.configurable()
def rmae_small(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, 
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = MultiscaleMaskedAutoencoderViT(**default_cfg)
    return model

@gin.configurable()
def rmae_base(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = MultiscaleMaskedAutoencoderViT(**default_cfg)
    return model

@gin.configurable()
def rmae_large(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = MultiscaleMaskedAutoencoderViT(**default_cfg)
    return model


if __name__=="__main__":
    x = torch.rand(10,3,224,224)
    model = rmae_tiny()
    model(x)