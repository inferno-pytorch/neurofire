

class Model(object):
    """ABC for models."""
    @classmethod
    def from_config(cls, **config):
        raise NotImplementedError
