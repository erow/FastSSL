import gin
from model.models_mae import *
import model.simclr
import model.simsiam
import model.moco
import model.sit

@gin.configurable()
def build_model(args,model_fn=mae_vit_base_patch16):
    model = model_fn()
    return model
