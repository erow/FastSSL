from typing import List
import gin

@gin.configurable()
class DynamicResolution:
    def __init__(self, scheme=gin.REQUIRED):
        self.scheme = scheme
    
    def get_config(self, epoch):
        for config in self.scheme:
            if epoch == config['epoch']:
                return config
        return None
    
    def __call__(self, model, loader, epoch,is_ffcv=False):        
        config = self.get_config(epoch)
        if config is None:
            # use the last config
            return
        print("Dynamic resolution: ", ", ".join([f"{k}={v}" for k,v in config.items()]))
        img_size = config['res']
        
        if is_ffcv:
            if not hasattr(loader, 'pipeline_specs'):
                raise AttributeError("FFCV loader does not have 'pipeline_specs' attribute")
            if 'image' not in loader.pipeline_specs:
                raise KeyError("FFCV loader does not have 'image' in pipeline_specs")
            pipeline = loader.pipeline_specs['image']
            if not hasattr(pipeline, 'decoder') or not hasattr(pipeline.decoder, 'output_size'):
                raise AttributeError("FFCV pipeline decoder does not have 'output_size' attribute")
            if pipeline.decoder.output_size[0] != img_size:
                pipeline.decoder.output_size = (img_size, img_size)
                if hasattr(loader, 'generate_code'):
                    loader.generate_code()
                else:
                    raise AttributeError("FFCV loader does not have 'generate_code' method")
        else:
            if not hasattr(loader, 'dataset'):
                raise AttributeError("DataLoader does not have 'dataset' attribute")
            if not hasattr(loader.dataset, 'transforms'):
                raise AttributeError(f"Dataset {type(loader.dataset).__name__} does not have 'transforms' attribute")
            transforms = loader.dataset.transforms
            if not hasattr(transforms, 'transform'):
                raise AttributeError("Dataset transforms does not have 'transform' attribute")
            augmentation = transforms.transform
            augmentation.size = (img_size,img_size)
                