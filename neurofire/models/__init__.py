from .cantor import Cantor


def get_model(name):
    assert name in globals(), "Model not found."
    return globals().get(name)
