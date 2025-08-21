from functools import partial
from typing import Final, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block,use_fused_attn,LayerScale, Attention, Mlp,DropPath

from timm.layers.pos_embed import resample_abs_pos_embed
from layers.operation import patchify
from util.pos_embed import get_2d_sincos_pos_embed
import gin
from layers.target import build_target

def select_tokens(x,index,dim=1):
    return torch.gather(x, dim=dim, index=index.unsqueeze(-1).repeat(1,1,x.shape[-1]))
class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int, dim_kv: Optional[int] = None,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()        
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if dim_kv is None:
            dim_kv = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, N, C = query.shape
        q = self.q(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            query = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            query = attn @ v

        query = query.transpose(1, 2).reshape(B, N, C)
        query = self.proj(query)
        query = self.proj_drop(query)
        return query

class DecoderBlock(nn.Module):
    def __init__(
            self,
            dim: int, dim_kv:int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sa = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ca = CrossAttention(
            dim, dim_kv,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim_kv)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = q + self.drop_path1(self.ls1(self.sa(self.norm1(q))))
        q = q + self.drop_path3(self.ls3(self.ca(self.norm3(q), self.norm4(kv))))
        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q
    
@gin.configurable()
class MultiScaleMAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone    
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 mask_ratio=0.75,
                 decoder_ratio=0.9,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, decoder_feature_size=None,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder_ratio = decoder_ratio
        self.embed_dim = embed_dim
        self.img_size=img_size
        self.norm_pix_loss = True
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, strict_img_size=False)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
                
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches , decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, embed_dim, 
                         decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if decoder_feature_size is None:
            decoder_feature_size = patch_size**2 * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_feature_size, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.        
        """
        N, C, H, W = x.shape  # batch, length, dim
        L = (H // self.patch_embed.patch_size[0]) * (W // self.patch_embed.patch_size[1])
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_enc = ids_shuffle[:, :len_keep]
        ids_pred =ids_shuffle[:, len_keep:]
        
        return ids_enc, ids_pred, ids_restore

    @torch.no_grad()
    def representation(self, x, pos_embed=None):
        B, C, H, W = x.shape
        ## dynamic pos embed
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            (H//self.patch_embed.patch_size[0], W//self.patch_embed.patch_size[1]),
        )
        # embed patches
        z =  self.forward_encoder(x, pos_embed=pos_embed)[-1]
        return z[:,1:].mean(1)

    def forward_encoder(self, x, ids_enc=None,pos_embed=None):
        if pos_embed is None:
            pos_embed = self.pos_embed
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]
        if ids_enc is not None:
            x = select_tokens(x, ids_enc)
        
        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        z = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i %3==2:
                z.append(x)
        return z
    
    def forward_decoder(self,x, kv_list, ids_vis, ids_pred):
        # embed patches
        x = self.patch_embed(x)
        x = self.decoder_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.decoder_pos_embed
        
        # replace with mask
        vis_x = select_tokens(x, dim=1, index=ids_vis)
        mask_token = self.mask_token.repeat(x.shape[0], self.decoder_pos_embed.shape[1], 1)
        mask_x = select_tokens(mask_token + self.decoder_pos_embed, dim=1, index=ids_pred)
        
        
        # append mask token
        x = torch.cat((vis_x, mask_x), dim=1)

        # apply Transformer blocks
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x, kv_list[i])

        # decoder prediction
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        return x
    
    def forward(self, imgs, return_variables=False,**kwargs):
        assert isinstance(imgs,list)
        x1,x2 = imgs[:2]
        ## dynamic pos embed
        self.log = {}
                
        # masking: length -> length * mask_ratio
        ids_enc, _, ids_restore = self.random_masking(x1, self.mask_ratio)
        
        # reference
        kv_list = self.forward_encoder(x1, ids_enc)

        # decoder
        ids_vis, ids_pred, ids_restore = self.random_masking(x2, self.decoder_ratio)
            
        pred = self.forward_decoder(x2,kv_list,ids_vis, ids_pred)
        target = patchify(x2, self.patch_embed.patch_size[0])
        
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        pred_c = pred[:,ids_vis.shape[1]:]
        target_c = select_tokens(target, dim=1, index=ids_pred)
        loss = F.mse_loss(pred_c, target_c, reduction='mean')
        
        self.log['loss'] = loss.item()
        return loss,self.log
        
@gin.configurable()
def msmae_tiny(**kwargs):
    model = MultiScaleMAE(
        patch_size=16,embed_dim=196,depth=12,num_heads=12,
        decoder_embed_dim=96,decoder_depth=1,decoder_num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@gin.configurable()
def msmae_small(**kwargs):
    model = MultiScaleMAE(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, 
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@gin.configurable()
def msmae_base(pretrained=None,**kwargs):
    model = MultiScaleMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained is not None:
        pass
    return model


@gin.configurable()
def msmae_large(**kwargs):
    model = MultiScaleMAE(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
