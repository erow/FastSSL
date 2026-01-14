"""
DiffMAE: Diffusion Models as Masked Autoencoders
Reference: https://arxiv.org/abs/2304.03283

DiffMAE combines masked autoencoding with diffusion models. Instead of directly predicting
masked patches, it learns to denoise gradually corrupted patches using a diffusion process.

Key components:
1. Standard ViT encoder (like MAE) that processes visible patches
2. Diffusion-based decoder that learns to denoise masked patches
3. Training with diffusion loss (predicting noise at various timesteps)

Training command:
```
torchrun --nproc_per_node=8 main_pretrain.py \
    --data_path=$FFCVTRAIN --data_set=ffcv \
    --epochs 1600 --warmup_epochs 40 \
    --opt adamw --opt_betas 0.9 0.95 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --batch_size 256 \
    --gin build_model.model_fn=@diffmae_base \
          build_dataset.transform_fn=@SimplePipeline \
    --ckpt_freq=100
```
"""

from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from timm.layers.pos_embed import resample_abs_pos_embed
from util.pos_embed import get_2d_sincos_pos_embed
import gin


class DiffusionSchedule:
    """
    Implements various noise schedules for diffusion models.
    Supports linear, cosine, and quadratic schedules.
    """
    def __init__(self, num_timesteps=1000, schedule='linear', beta_start=0.0001, beta_end=0.02,rho=0.8):
        self.num_timesteps = num_timesteps
        
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps) ** rho
        elif schedule == 'cosine':
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        elif schedule == 'quadratic':
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        # elif schedule == 'exponential':
        #     betas = torch.exp(torch.linspace(beta_start, beta_end, num_timesteps))
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Clamp alphas_cumprod to prevent numerical instability
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=1e-8, max=1.0-1e-8)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # Clamp before log to prevent NaN
        self.log_one_minus_alphas_cumprod = torch.log(torch.clamp(1.0 - self.alphas_cumprod, min=1e-8))
        # Add epsilon to prevent division by zero
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / (self.alphas_cumprod + 1e-8))
        # Clamp to prevent negative values under square root
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(torch.clamp(1.0 / self.alphas_cumprod - 1, min=0))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # Add epsilon to denominator to prevent division by very small numbers
        denominator = 1.0 - self.alphas_cumprod + 1e-8
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / denominator
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / denominator
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / denominator
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to x_start based on timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionDecoder(nn.Module):
    """
    Diffusion-based decoder for DiffMAE.
    Takes encoder output and noisy patches, predicts noise at timestep t.
    """
    def __init__(
        self,
        num_patches,
        encoder_embed_dim,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        patch_size=16,
        in_chans=3,
    ):
        super().__init__()
        
        self.decoder_embed_dim = decoder_embed_dim
        self.num_patches = num_patches
        
        # Project encoder output to decoder dimension
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.target_embed = nn.Linear(patch_size**2 * in_chans, decoder_embed_dim, bias=True)
        
        # Positional embedding for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        ## init with sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(num_patches**0.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Time step embedding (for diffusion timestep)
        self.time_embed = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.SiLU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
        )
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
    def forward(self, x, ids_restore, noisy_patches, timesteps, num_patches=None):
        """
        Args:
            x: Encoder output [N, L_visible, D_encoder]
            ids_restore: Indices to restore original order [N, L]
            noisy_patches: Noisy masked patches [N, L_masked, patch_dim]
            timesteps: Diffusion timesteps [N]
            num_patches: Number of patches (for dynamic resolution). If None, uses self.num_patches.
        
        Returns:
            Predicted noise [N, L_masked, patch_dim]
        """
        N = x.shape[0]
        
        # Determine actual number of patches (for dynamic resolution)
        # Total patches = visible + masked = ids_restore.shape[1]
        if num_patches is None:
            num_patches = ids_restore.shape[1]
        
        # Resample decoder positional embeddings if needed for dynamic resolution
        if num_patches != self.num_patches:
            # Calculate grid size from number of patches
            grid_size = int(num_patches ** 0.5)
            decoder_pos_embed = resample_abs_pos_embed(
                self.decoder_pos_embed,
                (grid_size, grid_size),
                num_prefix_tokens=1,
            )
        else:
            decoder_pos_embed = self.decoder_pos_embed
        
        # Embed encoder tokens
        x = self.decoder_embed(x)
        
        # Generate time embeddings
        t_emb = self.timestep_embedding(timesteps, self.decoder_embed_dim)
        t_emb = self.time_embed(t_emb).unsqueeze(1)  # [N, 1, D]
        
        target_emb = self.target_embed(noisy_patches)
        # Add time embedding to target patches
        target_emb = target_emb + t_emb
        
        # Concatenate visible tokens and target patches
        x_ = torch.cat([x[:, 1:, :], target_emb], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Add positional embeddings
        x = x + decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: 1-D tensor of N timesteps
            dim: Dimension of the embeddings
            max_period: Maximum period of the sinusoids
        
        Returns:
            [N, dim] tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class DiffusionMaskedAutoencoderViT(nn.Module):
    """
    Diffusion-based Masked Autoencoder with Vision Transformer backbone.
    """
    representation_names = ['avg', 'cls']
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        mask_ratio=0.75,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_diffusion_timesteps=1000,
        diffusion_schedule='linear',
        norm_pix_loss=True,
        pred_type='x',  # 'x' for x-prediction (default), 'eps' for epsilon prediction
        time_range=None,  # Tuple (min, max) to control timestep range, None for full range [0, num_diffusion_timesteps)
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, strict_img_size=False,dynamic_img_pad=True
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # Diffusion decoder specifics
        self.diffusion_decoder = DiffusionDecoder(
            num_patches=num_patches,
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
        
        # Noise predictor head
        self.noise_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )
        # --------------------------------------------------------------------------
        
        # Diffusion schedule
        self.diffusion_schedule = DiffusionSchedule(
            num_timesteps=num_diffusion_timesteps,
            schedule=diffusion_schedule,
        )
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.pred_type = pred_type
        
        # Time range for timestep sampling
        if time_range is None:
            self.time_range = (0, num_diffusion_timesteps)
        else:
            if not isinstance(time_range, (tuple, list)) or len(time_range) != 2:
                raise ValueError(f"time_range must be a tuple/list of length 2, got {time_range}")
            min_t, max_t = time_range
            if min_t < 0 or max_t > num_diffusion_timesteps or min_t >= max_t:
                raise ValueError(
                    f"time_range must satisfy 0 <= min < max <= num_diffusion_timesteps, "
                    f"got {time_range} with num_diffusion_timesteps={num_diffusion_timesteps}"
                )
            self.time_range = (min_t, max_t)
        
        self.initialize_weights()
        
    
    def initialize_weights(self):
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.diffusion_decoder.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True
        )
        self.diffusion_decoder.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )
        
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize tokens
        torch.nn.init.normal_(self.cls_token, std=0.02)
        # torch.nn.init.normal_(self.diffusion_decoder.mask_token, std=0.02)
        
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
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
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**0.5)
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
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        """
        Encode visible patches.
        """
        B, C, H, W = x.shape
        # Resample positional embeddings for dynamic image sizes
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            (H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]),
            num_prefix_tokens=1,
        )
        
        # Embed patches
        x = self.patch_embed(x)
        
        # Add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]
        
        # Masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_diffusion_decoder(self, latent, ids_restore, target_patches, mask):
        """
        Apply diffusion process to masked patches.
        
        Args:
            latent: Encoder output [N, L_visible, D]
            ids_restore: Indices to restore original order [N, L]
            target_patches: Ground truth patches [N, L, patch_dim]
            mask: Binary mask [N, L] (1 is masked, 0 is visible)
        
        Returns:
            Predicted noise and loss
        """
        N = latent.shape[0]
        
        # Get actual number of patches from ids_restore (for dynamic resolution)
        num_patches = ids_restore.shape[1]
        
        # Sample random timesteps for each sample in batch
        min_t, max_t = self.time_range
        timesteps = torch.randint(
            min_t, max_t, (N,), device=latent.device
        ).long()
        
        # Add noise to target patches according to diffusion schedule
        noise = torch.randn_like(target_patches)
        noisy_patches = self.diffusion_schedule.q_sample(
            target_patches, timesteps, noise=noise
        )
        
        # NaN detection: check for NaN in intermediate values
        if torch.isnan(noisy_patches).any():
            print(f"WARNING: NaN detected in noisy_patches! timesteps: {timesteps}, target_patches stats: min={target_patches.min()}, max={target_patches.max()}, mean={target_patches.mean()}")
        
        # Predict noise using decoder (pass num_patches for dynamic resolution)
        decoder_out = self.diffusion_decoder(
            latent, ids_restore, noisy_patches, timesteps, num_patches=num_patches
        )
        pred = self.noise_pred(decoder_out)
        
        # NaN detection: check for NaN in predictions
        if torch.isnan(pred).any():
            print(f"WARNING: NaN detected in pred! decoder_out stats: min={decoder_out.min()}, max={decoder_out.max()}, mean={decoder_out.mean()}")
        if torch.isnan(noise).any():
            print(f"WARNING: NaN detected in noise!")
        
        # Compute loss only on masked patches
        if self.pred_type == 'x':
            loss = (pred - target_patches) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        elif self.pred_type == 'eps':
            loss = (pred - noise) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            raise ValueError(f"Invalid prediction type: {self.pred_type}")
        
        return pred, loss
    
    def forward(self, imgs, **kwargs):
        """
        Forward pass for training.
        
        Args:
            imgs: Input images [N, 3, H, W]
        
        Returns:
            loss: Diffusion loss
            log: Dictionary of logging metrics
        """
        self.log = {}
        
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]
        
        # Patchify images to get ground truth
        target = self.patchify(imgs)
        
        # Normalize target if needed
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            # Clamp variance to prevent numerical instability in float16
            var = torch.clamp(var, min=1e-5)
            target = (target - mean) / torch.sqrt(var)
        
        # Encode visible patches
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)
        
        # Apply diffusion decoder
        pred_noise, loss = self.forward_diffusion_decoder(
            latent, ids_restore, target, mask
        )
        
        # NaN detection: check final loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf detected in final loss! loss={loss}, mask.sum()={mask.sum()}, target stats: min={target.min()}, max={target.max()}, mean={target.mean()}")
            # Log to self.log for monitoring
            self.log['nan_detected'] = 1.0
            self.log['loss_value'] = float('nan') if torch.isnan(loss) else float('inf')
        
        return loss, self.log
    
    def representation(self, x):
        """
        Extract representations for downstream tasks.
        """
        B, C, H, W = x.shape
        # Resample positional embeddings for dynamic image sizes
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            (H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]),
            num_prefix_tokens=1,
        )
        
        # Embed patches
        x = self.patch_embed(x)
        
        # Add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]
        
        # Append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Return average of patch tokens (or cls token)
        representation = {
            'avg': x[:, 1:].mean(1),
            'cls': x[:, 0, :],
        }
        return representation


PRETRAINED_WEIGHTS = {
    'mae_base': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
    'mae_large': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
    'mae_huge': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
}
# Model factory functions with different sizes

@gin.configurable()
def diffmae_tiny(pretrained=False, **kwargs):
    """DiffMAE Tiny: 5.7M parameters"""
    default_cfg = dict(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        decoder_embed_dim=96,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    default_cfg.update(kwargs)
    model = DiffusionMaskedAutoencoderViT(**default_cfg)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHTS['mae_tiny'], map_location='cpu', weights_only=True)
        print("loading mae weights to diffmae: ", model.load_state_dict(state_dict, strict=False))
    return model


@gin.configurable()
def diffmae_small(pretrained=False, **kwargs):
    """DiffMAE Small: 22M parameters"""
    default_cfg = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    default_cfg.update(kwargs)
    model = DiffusionMaskedAutoencoderViT(**default_cfg)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHTS['mae_small'], map_location='cpu', weights_only=True)
        print("loading mae weights to diffmae: ", model.load_state_dict(state_dict, strict=False))
    return model


@gin.configurable()
def diffmae_base(pretrained=False, **kwargs):
    """DiffMAE Base: 86M parameters"""
    default_cfg = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    default_cfg.update(kwargs)
    model = DiffusionMaskedAutoencoderViT(**default_cfg)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHTS['mae_base'], map_location='cpu', weights_only=True)
        state_dict = state_dict['model']
        print("loading mae weights to diffmae: ", model.load_state_dict(state_dict, strict=False))
    return model


@gin.configurable()
def diffmae_large(pretrained=False, **kwargs):
    """DiffMAE Large: 304M parameters"""
    default_cfg = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    default_cfg.update(kwargs)
    model = DiffusionMaskedAutoencoderViT(**default_cfg)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHTS['mae_large'], map_location='cpu', weights_only=True)
        state_dict = state_dict['model']
        print("loading mae weights to diffmae: ", model.load_state_dict(state_dict, strict=False))
    return model


@gin.configurable()
def diffmae_huge(pretrained=False, **kwargs):
    """DiffMAE Huge: 632M parameters"""
    default_cfg = dict(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=640,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    default_cfg.update(kwargs)
    model = DiffusionMaskedAutoencoderViT(**default_cfg)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHTS['mae_huge'], map_location='cpu', weights_only=True)
        state_dict = state_dict['model']
        print("loading mae weights to diffmae: ", model.load_state_dict(state_dict, strict=False))
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing DiffMAE models...")
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    
    # Test tiny model
    print("\nTesting diffmae_tiny...")
    model = diffmae_tiny()
    loss, log = model(x)
    print(f"Loss: {loss.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Test base model
    print("\nTesting diffmae_base...")
    model = diffmae_base()
    loss, log = model(x)
    print(f"Loss: {loss.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Test representation extraction
    print("\nTesting representation extraction...")
    repr = model.representation(x)
    print(f"Representation shape: {repr.shape}")
    
    print("\n✓ All tests passed!")

