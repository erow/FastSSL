"""
Reference: https://github.com/facebookresearch/mae

# Note
- no weight decay for bias
- adamw betas = 0.9 0.95

# Training
```python
torchrun --nproc_per_node=8 main_pretrain.py --data_path=$FFCVTRAIN --data_set=ffcv --epochs 800 --warmup_epochs 40 --opt adamw --opt_betas 0.9 0.95 --blr 1.5e-4 --weight_decay 0.05 --batch_size 512 --gin build_model.model_fn=@mae_small build_dataset.transform_fn=@SimplePipeline  --ckpt_freq=100 

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
# recipe https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md

@gin.configurable()
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone    
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 mask_ratio: float=0.75, 
                 ra=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, decoder_feature_size=None,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.ra = ra
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.img_size=img_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, strict_img_size=True)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

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
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

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

    def forward_encoder(self, x, mask_ratio:float,pos_embed=None):
        B, C, H, W = x.shape
        if pos_embed is None:
            pos_embed = self.pos_embed
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    
    
    def forward_decoder(self, x, ids_restore,pos_embed=None):
        if pos_embed is None:
            pos_embed = self.decoder_pos_embed
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

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
        if self.ra>1:
            imgs = imgs.repeat(self.ra,1,1,1)
        ## dynamic pos embed
        B, C, H, W = imgs.shape
        # pos_embed = resample_abs_pos_embed(
        #     self.pos_embed,
        #     (H//self.patch_embed.patch_size[0], W//self.patch_embed.patch_size[1]),
        # )
        # decoder_pos_embed = resample_abs_pos_embed(
        #     self.decoder_pos_embed,
        #     (H//self.patch_embed.patch_size[0], W//self.patch_embed.patch_size[1]),
        # )
        # pos_embed, decoder_pos_embed = self.pos_embed, self.decoder_embed
        ## dynamic pos embed        
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)        
        pred = self.forward_decoder(latent, ids_restore,)  # [N, L, p*p*3]        
        loss = self.target_loss(imgs, pred, mask)
        return loss, self.log

PRETRAINED_WEIGHTS = {
    'mae_base': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
    'mae_large': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
    'mae_huge': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
}

@gin.configurable()
def mae_tiny(pretrained=False, **kwargs):
    default_cfg = dict(
        patch_size=16,embed_dim=192,depth=12,num_heads=3,
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    default_cfg.update(kwargs)
        
    model = MaskedAutoencoderViT(**default_cfg)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHTS['mae_tiny'], map_location='cpu', weights_only=True)
        print("loading mae weights to mae: ", model.load_state_dict(state_dict, strict=False))
    return model

@gin.configurable() 
def mae_small(pretrained=False, **kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, 
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = MaskedAutoencoderViT(**default_cfg)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHTS['mae_small'], map_location='cpu', weights_only=True)
        print("loading mae weights to mae: ", model.load_state_dict(state_dict, strict=False))
    return model

@gin.configurable()
def mae_base(pretrained=False, **kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = MaskedAutoencoderViT(**default_cfg)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHTS['mae_base'], map_location='cpu', weights_only=True)
        print("loading mae weights to mae: ", model.load_state_dict(state_dict, strict=False))
    return model

@gin.configurable()
def mae_large(pretrained=False, **kwargs):
    default_cfg = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = MaskedAutoencoderViT(**default_cfg)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHTS['mae_large'], map_location='cpu', weights_only=True)
        print("loading mae weights to mae: ", model.load_state_dict(state_dict, strict=False))
    return model

@gin.configurable()
def mae_huge_patch14_dec512d8b(pretrained=False, **kwargs):
    default_cfg = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    default_cfg.update(kwargs)
        
    model = MaskedAutoencoderViT(**default_cfg)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHTS['mae_huge'], map_location='cpu', weights_only=True)
        print("loading mae weights to mae: ", model.load_state_dict(state_dict, strict=False))
    return model

if __name__ == '__main__':
    from timm.models.vision_transformer import Attention
    for dim in range(480, 384-1,-48):
        print(f'try dim={dim}',end=',',flush=True)
        x = torch.rand(512,197,dim).cuda()
        model = Attention(dim,1)
        model.train().cuda()
        with torch.amp.autocast('cuda'):
            latent = model(x)
            loss = latent.mean()
            loss.backward()
        
        print(f'success!')