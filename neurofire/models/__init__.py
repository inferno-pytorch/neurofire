from .cantor import Cantor
from .unet_2d import UNet2D


def get_model(name):
    assert name in globals(), "Model not found."
    return globals().get(name)
