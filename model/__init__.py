from .vgg.vgg import VGG_for_CAM
from .resnet.resnet import Resnet_for_CAM
from .googlenet.googlenet import Googlenet_for_CAM
from .alexnet.alexnet import Alexnet_for_CAM

__all__ = [
    VGG_for_CAM,
    Resnet_for_CAM,
    Googlenet_for_CAM,
    Alexnet_for_CAM,
]
