"""AIM: Autoregressive Image Models
Reference: https://github.com/apple/ml-aim

# Note:

## key points:
- Prefix causal attention: 
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gin
from huggingface_hub import PyTorchModelHubMixin

import torch
from torch import nn

import layers.aim_vit as layers
from layers.operation import patchify

__all__ = [
    "Transformer",
    "AIMPretrain",
    "AIMForImageClassification",
    "aim_600M",
    "aim_1B",
    "aim_3B",
    "aim_7B",
]

ArrayLike = Any
Module = Callable[..., Any]


class AIMMixin:
    preprocessor: Module
    trunk: Module
    head: Module

    def forward(
        self,
        x: ArrayLike,
        mask: Optional[ArrayLike] = None,
        max_block_id: Optional[int] = -1,
    ) -> ArrayLike:
        x = self.preprocessor(x)
        x, _ = self.trunk(x, mask=mask, max_block_id=max_block_id)
        logits = self.head(x)
        return logits

    def extract_features(
        self,
        x: ArrayLike,
        mask: Optional[ArrayLike] = None,
        max_block_id: Optional[int] = -1,
    ) -> List[ArrayLike]:
        x = self.preprocessor(x, mask=mask)
        feats = self.trunk(
            x, mask=mask, max_block_id=max_block_id, return_features=True
        )
        return feats

class Transformer(nn.Module):
    def __init__(
        self,
        attn_target: Callable[[bool], nn.Module],
        embed_dim: int,
        num_blocks: int,
        ffn_target: Callable[..., nn.Module] = layers.MLP,
        post_transformer_layer: Optional[nn.Module] = None,
        norm_layer: Callable[[int], nn.Module] = layers.LayerNorm,
        mlp_ratio: int = 4,
        mlp_hidden_dim: Optional[int] = None,
        ffn_dropout_rate: float = 0.0,
        use_bias: bool = False,
        post_trunk_norm: bool = True,
    ):
        super().__init__()
        if mlp_hidden_dim is None:
            mlp_hidden_dim = int(mlp_ratio * embed_dim)

        self.blocks = nn.ModuleList(
            [
                layers.Block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    ffn_target=ffn_target,
                    mlp_hidden_dim=mlp_hidden_dim,
                    norm_layer=norm_layer,
                    ffn_dropout_rate=ffn_dropout_rate,
                    use_bias=use_bias,
                )
                for _ in range(num_blocks)
            ]
        )
        self.post_trunk_norm = norm_layer(embed_dim) if post_trunk_norm else None
        self.post_transformer_layer = post_transformer_layer

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        max_block_id: Optional[int] = -1,
        return_features: bool = False,
    ) -> Union[Tuple[torch.Tensor, List[torch.Tensor]], List[torch.Tensor]]:
        # only evaluate up to the max block id
        if max_block_id is None:
            assert (
                self.post_transformer_layer is not None
            ), "Unable to determine the max block id."
            max_block_id = self.post_transformer_layer.max_block_id

        features = []
        for blk_id, blk in enumerate(self.blocks):
            tokens = blk(tokens, mask=mask)
            features.append(tokens)

            if blk_id == max_block_id:
                break

        if return_features:
            return features

        if self.post_trunk_norm is not None:
            tokens = self.post_trunk_norm(tokens)

        if self.post_transformer_layer is not None:
            tokens = self.post_transformer_layer(tokens, layer_features=features)

        return tokens, features



@gin.configurable()
class AIMPretrain(nn.Module):
    def __init__(self, preprocessor: nn.Module, trunk: nn.Module, head: layers.ReconstructionHead,
                 norm_pix_loss=True, prefix_len=16,
                 ):
        super().__init__()
        self.preprocessor = preprocessor
        self.trunk = trunk
        self.head = head
        self.norm_pix_loss = norm_pix_loss
        self.prefix_len = prefix_len
        

    def forward(
        self,
        imgs: ArrayLike,
        mask: Optional[ArrayLike] = None,
        max_block_id: Optional[int] = -1,
        **kwargs: Any,
    ) -> ArrayLike:
        
        x = self.preprocessor(imgs)
        L = x.shape[1]
        if mask is None:
            mask = torch.ones(L, L, dtype=torch.bool,device=x.device).tril(diagonal=0)
            mask[:, :self.prefix_len] = True # set the first prefix_len tokens to be visible
            
        x, _ = self.trunk(x, mask=mask, max_block_id=max_block_id)
        pred = self.head(x)
        
        # compute loss
        target = patchify(imgs,self.head.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred[:,self.prefix_len:-1] - target[:,self.prefix_len+1:]) ** 2 # next token prediction
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.mean()
        return loss, {}
    
    
    def extract_features(
        self,
        x: ArrayLike,
        mask: Optional[ArrayLike] = None,
        max_block_id: Optional[int] = -1,
    ) -> List[ArrayLike]:
        x = self.preprocessor(x, mask=mask)
        feats = self.trunk(
            x, mask=mask, max_block_id=max_block_id, return_features=True
        )
        return feats
    
    @torch.no_grad()
    def representation(self, x: ArrayLike, **kwargs) -> ArrayLike:
        feats = self.extract_features(x, mask=None, max_block_id=-1)
        z = feats[-1]
        return dict(latent=z[:,-1])

class AIMForImageClassification(AIMMixin, PyTorchModelHubMixin, nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.preprocessor, self.trunk, self.head = aim_config(**config)
        
        


def _get_attention_target(dim: int, num_heads: int) -> Callable[[bool], nn.Module]:
    def callback(use_bias: bool) -> nn.Module:
        return layers.Attention(dim=dim, num_heads=num_heads, use_bias=use_bias)

    return callback

@gin.configurable()
def aim_config(
    img_size: Union[int, Tuple[int, int]],
    patch_size: Union[int, Tuple[int, int]],
    embed_dim: int,
    num_blocks: int,
    num_heads: int,
    num_channels: int = 3,
    probe_layers: Union[int, Tuple[int, ...]] = 6,
    num_classes: int = 1000,
    mode = 'reconstruction',
    **kwargs: Any,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    # common
    norm_layer = layers.LayerNorm

    # preprocessor
    patchifier = layers.PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=num_channels,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
    )
    preprocessor = layers.ViTPreprocessor(
        patchifier, drop_patches=False, cls_token=False
    )

    # trunk
    if isinstance(probe_layers, int):
        probe_layers = tuple(range(num_blocks - probe_layers, num_blocks))
    assert all(layer >= 0 for layer in probe_layers), probe_layers

    attn_target = _get_attention_target(dim=embed_dim, num_heads=num_heads)
    post_transform_layer = layers.AverageLayers(probe_layers, reduce=False)
    trunk = Transformer(
        attn_target,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        norm_layer=norm_layer,
        post_transformer_layer=post_transform_layer,
        **kwargs,
    )

    # head
    if mode == 'classification':
        head = layers.AttentionPoolingClassifier(
            dim=embed_dim,
            out_features=num_classes,
            num_heads=num_heads,
            qkv_bias=False,
            num_queries=1,
        )
    elif mode == 'reconstruction':
        head = layers.ReconstructionHead(
            dim = embed_dim,
            patch_size = patch_size,
        )
    return preprocessor, trunk, head

@gin.configurable()
def aim_tiny(img_size: Union[int, Tuple[int, int]] = 32, **kwargs: Any) -> AIMPretrain:
    preprocessor, trunk, head = aim_config(
        img_size=img_size,
        patch_size=4,
        embed_dim=192,
        num_blocks=12,
        num_heads=12,
        **kwargs,
    )
    return AIMPretrain(preprocessor, trunk, head)

@gin.configurable()
def aim_600M(img_size: Union[int, Tuple[int, int]] = 224, **kwargs: Any) -> AIMPretrain:
    preprocessor, trunk, head = aim_config(
        img_size=img_size,
        patch_size=14,
        embed_dim=1536,
        num_blocks=24,
        num_heads=12,
        **kwargs,
    )
    return AIMPretrain(preprocessor, trunk, head)


def aim_1B(img_size: Union[int, Tuple[int, int]] = 224, **kwargs: Any) -> AIMPretrain:
    preprocessor, trunk, head = aim_config(
        img_size=img_size,
        patch_size=14,
        embed_dim=2048,
        num_blocks=24,
        num_heads=16,
        **kwargs,
    )
    return AIMPretrain(preprocessor, trunk, head)


def aim_3B(
    img_size: Union[int, Tuple[int, int]] = 224, patch_size: int = 14, **kwargs: Any
) -> AIMPretrain:
    preprocessor, trunk, head = aim_config(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=3072,
        num_blocks=24,
        num_heads=24,
        **kwargs,
    )
    return AIMPretrain(preprocessor, trunk, head)


def aim_7B(
    img_size: Union[int, Tuple[int, int]] = 224, patch_size: int = 14, **kwargs: Any
) -> AIMPretrain:
    preprocessor, trunk, head = aim_config(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=4096,
        num_blocks=32,
        num_heads=32,
        **kwargs,
    )
    return AIMPretrain(preprocessor, trunk, head)



if __name__ == '__main__':
    import torch.nn.functional as F
    model = aim_tiny(num_classes=10)
    params = sum(p.numel() for p in model.parameters())
    x = torch.randn(10, 3, 32, 32)
    out = model(x)
    print(f"AIM: Autoregressive Image Models, size = {params:_}")
    # print(out.shape)
    