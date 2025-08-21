"""SplitMask model implementation.
Paper: https://arxiv.org/pdf/2112.10740

# Note:

## key points:
- denoising autoencoders: 
- Learning using non object-centric images
"""

from copy import deepcopy
import gin
import torch
from torch import nn
import torch.nn.functional as F
from layers.mae import *
from layers.operation import build_head, contrastive_loss


class SplitMask(nn.Module):
    def __init__(self, 
                 out_dim = 1024, T=1.0,
                 img_size=224, patch_size=16, in_chans=3,                 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.T = T
        
        self.encoder = VisionTransformer(
            num_classes=0,global_pool='avg',
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer
        )
        
        self.decoder = MaskedDecoderViT(
            embed_dim,
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
            decoder_feature_size=embed_dim, mlp_ratio=mlp_ratio, norm_layer=norm_layer
        )
        
        self.num_prefix_tokens = self.encoder.num_prefix_tokens
        self.num_patches = self.encoder.patch_embed.num_patches
        
        
        self._teacher = deepcopy(self.encoder)
        self._teacher.requires_grad_(False)
        self.update(0)

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4), nn.ReLU(),
            nn.BatchNorm1d(embed_dim*4), 
            nn.Linear(embed_dim*4, out_dim,bias=False)
        )
        self.bn = nn.BatchNorm1d(embed_dim,affine=False)
        self.predictor = build_head(2,out_dim,embed_dim*4,out_dim,False)
    
    @torch.no_grad()
    def teacher(self,x):
        return self._teacher.forward_features(x)
    
    @torch.no_grad()
    def update(self,m):
        for param_b, param_m in zip(self.encoder.parameters(), self._teacher.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
            
    @torch.no_grad()
    def representation(self, x):
        x = self.encoder(x)
        return x

    def tokenizer(self,x):
        # tokenization
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.patch_drop(x)
        return x
    
    def forward_encoder(self, x):
        x = self.encoder.norm_pre(x)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x
    
    def forward_decoder(self, x_pre, x_pat, mask):
        decoder = self.decoder
        # embed tokens
        pos_embed = decoder.decoder_pos_embed
        x_pre = decoder.decoder_embed(x_pre)
        x_pat = decoder.decoder_embed(x_pat)

        B, L = mask.shape
        # append mask tokens to sequence
        mask_tokens = decoder.mask_token.repeat(B, L - x_pat.shape[1], 1)
        x_ = torch.cat([x_pat, mask_tokens], dim=1)  # no cls token        
        # add pos embed
        x_ = x_ + pos_embed # no pos embedding for cls tokens
        
        x = torch.cat([x_pre, x_], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in decoder.decoder_blocks:
            x = blk(x)
        x = decoder.decoder_norm(x)

        # predictor projection
        x = decoder.decoder_pred(x)

        # remove cls token
        x = x[:, -L:, :]

        return x
    
    def forward(self, imgs,**kwargs):
        
        tokens = self.tokenizer(imgs)
                
        # masking: length -> length * mask_ratio
        x_pre,x_pat = tokens.split([self.num_prefix_tokens, self.num_patches], dim=1)
        
        B,L,C = x_pat.shape
        # split mask patches
        noise = torch.rand(B, L, device=x_pat.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        x_shuffle = torch.gather(x_pat, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        x1, x2 = x_shuffle.chunk(2,dim=1)
        x1 = torch.cat([x_pre, x1], dim=1)
        x2 = torch.cat([x_pre, x2], dim=1)
        
        x1 = self.forward_encoder(x1)
        x2 = self.forward_encoder(x2)
        
        ################## CL for latent variables ##################
        x1_pre, x1_pat = x1[:,:self.num_prefix_tokens], x1[:,self.num_prefix_tokens:]
        x2_pre, x2_pat = x2[:,:self.num_prefix_tokens], x2[:,self.num_prefix_tokens:]
        
        z1, z2 = x1_pat.mean(1), x2_pat.mean(1)
        z1, z2 = self.projector(z1), self.projector(z2)
        q1, q2 = self.predictor(z1), self.predictor(z2)
        with torch.no_grad():
            k_pre,k_pat = self.teacher(imgs).split([self.num_prefix_tokens, self.num_patches], dim=1)
            k = k_pat.mean(1)
            k = self.projector(k)
            
        cl_loss = ( contrastive_loss(q1, k, self.T) + 
                contrastive_loss(q2, k, self.T))/2
        
        
        ################## decoder ##################
        
        pred1 = self.decoder(x1_pre,x1_pat, ids_restore)
        ids1,ids2 = ids_restore.chunk(2,1)
        ids_restore2 = torch.cat([ids2,ids1],1)
        pred2 = self.decoder(x2_pre,x2_pat, ids_restore2)
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x1_pat.device)
        mask[:, :L//2] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        pred = torch.where(mask.unsqueeze(-1).repeat(1,1,C)>0,pred1,pred2)
        
        pred_loss = F.mse_loss(pred, (k_pat))
        
        loss = cl_loss + pred_loss
        self.log = {
            'cl_loss': cl_loss.item(),
            'pred_loss': pred_loss.item()
        }
        return loss, self.log
    
@gin.configurable
def tiny_split(**kwargs):
    model = SplitMask(
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=12,
        decoder_embed_dim=96, decoder_depth=1, decoder_num_heads=3,
        **kwargs
    )
    return model

if __name__=='__main__':
    model = SplitMask()
    x = torch.rand(2,3,224,224)
    print(model(x))