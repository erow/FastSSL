import math
import einops
import gin
import torch
from torch import nn
from .operation import patchify, unpatchify


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


def get_gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""

    def _gaussian_fn(kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n**2)
        return w

    gkern1d = _gaussian_fn(kernlen, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()

class HOGLayerC(nn.Module):
    # copy from https://github.com/facebookresearch/SlowFast/blob/2efb99faa254075b4e28d3d4f313052b51da05bc/slowfast/models/operators.py#L66
    def __init__(self, nbins=9, pool=7, gaussian_window=0):
        super(HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer("gkern", gkern)

    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=3
        )
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=3
        )
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros(
            (b, c, self.nbins, h, w), dtype=torch.float, device=x.device
        )
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, "h {} gw {}".format(
                    h, self.gaussian_window
                )
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = torch.nn.functional.normalize(out, p=2, dim=2)

        return out  # B 3 nbins H W            
@gin.configurable
class TargetHOG(nn.Module):
    def __init__(self, patch_size=16,ignore_mask=False,pool=4):
        super().__init__()
        self.patch_size = patch_size
        self.ignore_mask = ignore_mask
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
