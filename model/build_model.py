import gin
from model.models_mae import *

@gin.configurable()
def build_model(args,model_fn=mae_vit_base_patch16):
    if args.flash_attn:
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
        import torch.nn.functional as F
        F.scaled_dot_product_attention = flash_attn_func
    model = model_fn()
    return model
