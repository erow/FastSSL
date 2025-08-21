import gin
import timm

@gin.configurable
def create_backbone(name='resnet50', num_classes=0, **kwargs):
    return timm.create_model(name,  num_classes=num_classes,**kwargs)