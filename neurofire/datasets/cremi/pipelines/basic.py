from ..loaders import get_cremi_loaders
from ....models import get_model
from inferno.trainers.basic import Trainer
from inferno.utils.io_utils import yaml2dict


class BasicPipeline(object):
    def __init__(self):
        self._trainer = None
        self._model = None

    @property
    def trainer(self):
        assert isinstance(self._trainer, Trainer)
        return self._trainer

    def build_trainer(self, **config):
        self._trainer = Trainer.build(self._model, **config)
        return self

    def build_model(self, **config):
        model = get_model(config.pop('model_name')).from_config(**config)
        self._model = model
        return self

    def build_loaders(self, **config):
        train_loader_config = config.get('train')
        validate_loader_config = config.get('validate')
        if train_loader_config is not None:
            train_loader = get_cremi_loaders(train_loader_config)
            self.trainer.bind_loader('train', train_loader)
        if validate_loader_config is not None:
            validate_loader = get_cremi_loaders(validate_loader_config)
            self.trainer.bind_loader('validate', validate_loader)
        return self

    def train(self):
        self.trainer.fit()

    @classmethod
    def build(cls, config):
        config = yaml2dict(config)
        trainer_config = yaml2dict(config.get('trainer_config'))
        model_config = yaml2dict(config.get('model_config'))
        data_config = yaml2dict(config.get('data_config'))
        # Build the pipeline
        pipeline = cls() \
            .build_model(**model_config) \
            .build_trainer(**trainer_config)\
            .build_loaders(**data_config)
        return pipeline
