import einops
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import minmax_scale


def show_images(images, title=None,nrow=8,):
    """Display a list of images."""
    from torchvision.utils import make_grid
    grid = make_grid(images, nrow=nrow, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    if title:
        plt.title(title)
        

def pca_components(z, offset=0,n_components=10,threshold=-1):
    """ the first n_components of PCA.
    :param z: (B, C, H, W)"""
    
    B, C, H, W = z.shape
    # z = F.normalize(z, dim=-1)
    
    z = z.permute(0, 2, 3, 1).flatten(0, -2).cpu().numpy()  # (B*H*W, C)
    pca = PCA(n_components=n_components)
    
    feat = pca.fit_transform(z)
    fg_mask = minmax_scale(feat[:, 0], feature_range=(0, 1)) > threshold
    
    pca_features = np.zeros((B*H*W, n_components))
    
    
    # feat = pca.fit_transform(z.cpu().numpy()*fg_mask[:, None])  # reapply PCA to the masked features

    feat = pca.fit_transform(z[fg_mask])
    # normalize
    feat = minmax_scale(feat, feature_range=(0, 1))
    
    
    pca_features[fg_mask] = feat
    pca_features[~fg_mask] = 0  # set the rest to zero
    
    pca_features = pca_features.reshape(-1, H, W, n_components)  # (B*H*W, n_components) -> (B, H, W, n_components
    pca_features = torch.tensor(pca_features).permute(0, 3, 1, 2)[:,offset:offset+3]  # (B, n_components, H, W)
    # feat = F.interpolate(feat, size=(112,112), mode='bilinear', align_corners=False)
   
    return pca_features
