import torch.nn as nn


class Server(object):
    def __init__(self, model=None):
        # Privates
        self._model = None
        # Set model if required
        if model is not None:
            self.model = model

    @property
    def model(self):
        assert self._model is not None, "Model is not defined."
        return self._model

    @model.setter
    def model(self, value):
        assert isinstance(value, nn.Module), "Object to set to model is not a torch.nn.Module."
        self._model = value

    def bind_model(self, model):
        self.model = model
        return self
