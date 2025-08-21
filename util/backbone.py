import torch, os
import timm
from torch import nn
import gin

@gin.configurable
def build_backbone(name: str, pretrained: bool = True, **kwargs):
    """
    Build a backbone model using timm library.

    Args:
        name (str): Name of the backbone model.
        pretrained (bool): Whether to load pretrained weights.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        torch.nn.Module: The backbone model.
    """
    if 'dinov3' in name:
        dino_root = os.environ.get('DINO3_PATH', '/models/dinov3')
        model = torch.hub.load(
            os.path.join(dino_root,'code'), name, 
            weights=os.path.join(dino_root,'models', f'{name}_pretrain_lvd1689m.pth'),
             source='local')
        model.norm = nn.Identity()

    else:
        model = timm.create_model(name, pretrained=pretrained, **kwargs)
    
    return model
    