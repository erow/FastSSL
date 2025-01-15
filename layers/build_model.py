import gin
from timm.layers import convert_sync_batchnorm
from model.mae import *

@gin.configurable()
def build_model(args, model_fn=mae_vit_base_patch16, **kwargs):
    model = model_fn(**kwargs)    
    model = convert_sync_batchnorm(model)
    return model
