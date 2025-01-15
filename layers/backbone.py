import gin
import timm

@gin.configurable
def create_backbone(name='resnet50', **kwargs):
    return timm.create_model(name, num_classes=0, **kwargs)