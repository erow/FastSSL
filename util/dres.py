from typing import List
import gin


DEFAULT_SCHEME ={
    1: [
        dict(res=160,mask_ratio=0.5,),
        dict(res=192,mask_ratio=0.66),
        dict(res=224,mask_ratio=0.75),
    ],
    2: [
        dict(res=160,mask_ratio=0.75,),
        dict(res=192,mask_ratio=0.75),
        dict(res=224,mask_ratio=0.75),
    ],
    3: [
        dict(res=224,mask_ratio=0.75),
        dict(res=192,mask_ratio=0.75),
        dict(res=160,mask_ratio=0.75),
    ],
    4: [
        dict(res=160,mask_ratio=0.75,),
        dict(res=192,mask_ratio=0.80),
        dict(res=224,mask_ratio=0.85),
    ],
    5: [
        dict(res=224,mask_ratio=0.75),
        dict(res=192,mask_ratio=0.75),
        dict(res=224,mask_ratio=0.75),
    ],
    6: [
        dict(res=224,mask_ratio=0.75),
        dict(res=192,mask_ratio=0.75),
        dict(res=224,mask_ratio=0.85),
    ],
}

@gin.configurable
class DynamicMasking:
    def __init__(self, start_ramp=gin.REQUIRED, end_ramp=gin.REQUIRED,  
                    scheme = 2):
        if isinstance(scheme, int):
            scheme = DEFAULT_SCHEME[scheme]
        else:
            assert isinstance(scheme, list)            
        self.scheme = scheme
        self.start_ramp = start_ramp
        self.end_ramp = end_ramp
    
    def get_config(self, epoch):
        if epoch <= self.start_ramp:
            return self.scheme[0]
        elif epoch>=self.end_ramp:
            return self.scheme[-1]
        else:
            i = (epoch-self.start_ramp) * (len(self.scheme)-1) // (self.end_ramp-self.start_ramp)
            return self.scheme[i]
    
    def __call__(self, model, loader, epoch,is_ffcv=False):        
        config = self.get_config(epoch)
        print(", ".join([f"{k}={v}" for k,v in config.items()]))
        img_size = config['res']
        mask_ratio = config['mask_ratio']
        
        assert hasattr(model,"mask_ratio")
        model.mask_ratio = mask_ratio
        if is_ffcv:
            pipeline=loader.pipeline_specs['image']
            if pipeline.decoder.output_size[0] != img_size:
                pipeline.decoder.output_size = (img_size,img_size)
                loader.generate_code()
        else:
            print(loader.dataset.transforms)
            augmentation = loader.dataset.transforms.transform
            augmentation.change_resolution(img_size)
                