"""DiFormer is a transformer decoder model for efficient representation learning.
It is designed to learn representations by predicting the difference between two images. Specifically, it uses a backbone model to extract features from the small version of images, then applies a transformer decoder to learn the difference between these features.

train:

torchrun main_pretrain.py  --batch_size=512 --opt adamw --opt_betas 0.9 0.95 --blr=5e-4 --weight_decay=0.05 --epochs=400 --warmup_epochs=10 --ckpt_freq=20   --data_path=$FFCVTRAIN --data_set=ffcv --gin build_dataset.transform_fn=@SimplePipeline build_model.model_fn=@diformer_base 
"""

from functools import partial

try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    rearrange = None
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block, Mlp

from timm.layers.pos_embed import resample_abs_pos_embed
from util.pos_embed import get_2d_sincos_pos_embed
import gin
from torchvision.transforms import GaussianBlur
from layers.target import build_target
# recipe https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md


from util.backbone import build_backbone

def normalize_token(tokens):
    mean = tokens.mean(dim=-1, keepdim=True)
    std = tokens.std(dim=-1, keepdim=True)
        
    tokens = (tokens - mean) / (std + 1e-6)
    return tokens

class DiFormer(nn.Module):
    """
    """
    def __init__(self, 
                 model_name = 'dinov3_vitb16',
                 small_scale=0.5, 
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 use_predictor=False,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.embed_dim = embed_dim
        self.small_scale = small_scale
        self.img_size=img_size
        
        reference = build_backbone(model_name)
        reference.requires_grad_(False).eval().cuda()
        self.reference = [reference] # a trick to prevent from being registered as a module
        # --------------------------------------------------------------------------
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, strict_img_size=True)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads, 
                                                   dim_feedforward=embed_dim * mlp_ratio, 
                                                   dropout=0.1,
                                                   activation='gelu',
                                                   batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = norm_layer(embed_dim)
        
        if use_predictor:
            self.predictor = Mlp(embed_dim, embed_dim*mlp_ratio, embed_dim, act_layer=nn.GELU, drop=0.0)
        else:
            self.predictor = nn.Identity()
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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
        output = self.reference[0].forward_features(x)
        z_local = output['x_norm_patchtokens'] # B, N, D 
        z_global = output['x_norm_clstoken']
        return z_local,z_global 

    def forward_decoder(self, x, kv, pos_embed=None):
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
        x = self.decoder(x, kv)
        x = self.norm(x)
        x = self.predictor(x)
        return x


    def forward(self, imgs,**kwargs):
        self.log = {}
        xl = imgs
        xs = F.interpolate(xl, scale_factor=self.small_scale, mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            zl_local,zl_global = self.forward_encoder(xl)
            zs_local,zs_global = self.forward_encoder(xs)
            H = int(zl_local.shape[1]**.5)
            h = int(zs_local.shape[1]**.5)
            # interpolate zs local
            if EINOPS_AVAILABLE:
                zs_local_up = rearrange(zs_local, 'b (h w) d -> b d h w', h=h)
                zs_local_up = F.interpolate(zs_local_up, size=(H, H), mode='bilinear', align_corners=False)
                zs_local_up = rearrange(zs_local_up, 'b d h w -> b (h w) d')
            else:
                # Fallback without einops
                zs_local_up = zs_local.view(B, h, h, -1).permute(0, 3, 1, 2)
                zs_local_up = F.interpolate(zs_local_up, size=(H, H), mode='bilinear', align_corners=False)
                zs_local_up = zs_local_up.permute(0, 2, 3, 1).view(B, H*H, -1)

        global_diff = zl_global - zs_global
        local_diff = zs_local_up - zl_local
        
        # normalize
        # global_diff = normalize_token(global_diff)
        # local_diff = normalize_token(local_diff)
        

        pred = self.forward_decoder(xl, zs_local)
        
        pred_local = pred[:, 1:]  # remove cls token
        pred_global = pred[:, 0]  # cls token
        
        
        loss_local = F.smooth_l1_loss(pred_local, local_diff, beta=0.1)
        loss_global = F.smooth_l1_loss(pred_global, global_diff, beta=0.1)
        loss = loss_local + loss_global
        self.log['loss_local'] = loss_local.item()
        self.log['loss_global'] = loss_global.item()
        return loss, self.log

@gin.configurable()
def diformer_tiny(**kwargs):
    default_cfg = dict(
        patch_size=16,embed_dim=192,depth=6,num_heads=3,
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    default_cfg.update(kwargs)
        
    model = DiFormer(**default_cfg)
    return model

@gin.configurable()
def diformer_small(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=384, depth=6, num_heads=6, 
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = DiFormer(**default_cfg)
    return model

@gin.configurable()
def diformer_base(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=768, depth=6, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = DiFormer(**default_cfg)
    return model

@gin.configurable()
def diformer_large(**kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = DiFormer(**default_cfg)
    return model

@gin.configurable()
def diformer_huge_patch14_dec512d8b(**kwargs):
    default_cfg = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = DiFormer(**default_cfg)
    return model

if __name__ == '__main__':
    from timm.models.vision_transformer import Attention
    x1 = torch.randn(2,3,224,224)
    x2 = torch.randn(2,3,224,224)
    
    model = diformer_base()
    model(x1)