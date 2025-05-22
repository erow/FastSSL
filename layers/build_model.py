import gin
from timm.layers import convert_sync_batchnorm
import os
import importlib.util
import glob

custom_model_files = glob.glob("model/*.py")
for file in custom_model_files:
    module_name = os.path.splitext(os.path.basename(file))[0]
    spec = importlib.util.spec_from_file_location(module_name, file)
    module = importlib.util.module_from_spec(spec)      
    spec.loader.load_module(module_name)
    
@gin.configurable()
def build_model(args, model_fn=gin.REQUIRED, **kwargs):
    model = model_fn(**kwargs)    
    model = convert_sync_batchnorm(model)
    return model
