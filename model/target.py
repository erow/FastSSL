import einops
import gin
import torch
from torch import nn
from .operation import patchify,unpatchify


@gin.configurable
class TargetMSE(nn.Module):
    def __init__(self,norm_pix_loss=True,patch_size=16,ignore_mask=False):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.ignore_mask = ignore_mask
    
    def forward(self, imgs, pred,mask=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if len(pred.shape)==4:
            pred = patchify(pred,self.patch_size)
        
        target = patchify(imgs,self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if mask is None or self.ignore_mask:
            loss = loss.mean()
        else:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

@gin.configurable
class TargetSSIM(nn.Module):
    def __init__(self,patch_size=16,ignore_mask=False):
        super().__init__()
        self.patch_size = patch_size
        self.ignore_mask = ignore_mask
        from torchmetrics.image import ssim
        self.ssim_loss = ssim.StructuralSimilarityIndexMeasure()
    
    def forward(self, imgs, pred,mask=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, 3, H, W]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if len(pred.shape)==3:
            pred = unpatchify(pred,self.patch_size)
        
        target = imgs      
        if mask is None or self.ignore_mask:
            loss = self.ssim_loss(pred,target)
        else:
            mask = mask.unsqueeze(-1).expand(-1,-1,3 * self.patch_size**2)
            mask = unpatchify(mask,self.patch_size)            
            loss = self.ssim_loss(pred*mask,target*mask)
        return 1 - loss
            
@gin.configurable
class TargetHOG(nn.Module):
    def __init__(self, patch_size=16,ignore_mask=False,pool=4):
        super().__init__()
        self.patch_size = patch_size
        self.ignore_mask = ignore_mask
        from .operation import HOGLayerC
        self.hog = HOGLayerC(pool=pool)
        
        self.feat_size = (patch_size//pool)
    
    def forward(self, imgs, pred,mask=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if len(pred.shape)==4:
            pred = patchify(pred,self.patch_size)
        
        hog_feat = self.hog(imgs) # [N, 3, Orientation, H, W]
        hog_feat = einops.rearrange(hog_feat,'n l c (h p1) (w p2) -> n (h w) (l c p1 p2)',p1=self.feat_size,p2=self.feat_size) # [N,L, C]
        
        loss = (pred - hog_feat) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if mask is None or self.ignore_mask:
            loss = loss.mean()
        else:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

@gin.configurable
def build_target(*args,target_fn=TargetMSE,**kwargs):
    return target_fn(*args,**kwargs)
