import gin
from model.models_mae import mae_vit_small_patch16
import model.simclr
import model.simsiam
import model.moco
import model.sit
import model.dino

@gin.configurable()
def build_model(args,model_fn=mae_vit_small_patch16):
    model = model_fn()
    return model
