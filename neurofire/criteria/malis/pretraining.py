import torch
import dill
from os.path import join
from inferno.trainers.callbacks.base import Callback
from inferno.utils.train_utils import Frequency


class Pretrain(Callback):
    def __init__(self, duration=None, save_pretrained_model=True):
        super(Pretrain, self).__init__()
        # Privates
        self._duration = None
        self._pretraining_criterion = {}
        self._training_criterion = {}
        self._pretraining_optimizer = {}
        self._training_optimizer = {}
        self._is_pretraining = True
        self._save_pretrained_model = save_pretrained_model
        # Publics
        if duration is not None:
            self.duration = duration

    @property
    def duration(self):
        assert self._duration is not None, "Pretraining duration is not defined yet."
        return self._duration.value, self._duration.units

    @duration.setter
    def duration(self, value):
        self._duration = Frequency.build_from(value)

    @property
    def is_pretraining(self):
        return self._is_pretraining

    def set_pretraining_criterion(self, method, **kwargs):
        self._pretraining_criterion.update({'method': method})
        self._pretraining_criterion.update(kwargs)
        return self

    def set_training_criterion(self, method, **kwargs):
        self._training_criterion.update({'method': method})
        self._training_criterion.update(kwargs)
        return self

    def set_pretraining_optimizer(self, method, param_groups=None, **kwargs):
        self._pretraining_optimizer.update({'method': method, 'param_groups': param_groups})
        self._pretraining_optimizer.update(kwargs)
        return self

    def set_training_optimizer(self, method, param_groups=None, **kwargs):
        self._training_optimizer.update({'method': method, 'param_groups': param_groups})
        self._training_optimizer.update(kwargs)
        return self

    def begin_of_fit(self, **_):
        self.trainer.build_criterion(**self._pretraining_criterion)
        self.trainer.build_optimizer(**self._pretraining_optimizer)

    def save_pretrained_model(self):
        if self._save_pretrained_model:
            torch.save(self.trainer.model, join(self.trainer.save_directory, 'pretrained.pytorch'),
                       pickle_module=dill)

    def begin_of_training_iteration(self, **_):
        # Check if pretraining time is up
        stopped_pretraining_this_iteration = False
        if self._is_pretraining:
            # If the following is false once, we never enter this branch again
            self._is_pretraining = \
                not self._duration.match(iteration_count=self.trainer.iteration_count,
                                         epoch_count=self.trainer.epoch_count)
            stopped_pretraining_this_iteration = not self._is_pretraining

        if stopped_pretraining_this_iteration:
            self.save_pretrained_model()
            self.trainer.print("Pretraining done. Building criterion and optimizer for training.")
            self.trainer.build_criterion(**self._training_criterion)
            self.trainer.build_optimizer(**self._training_optimizer)

    def begin_of_epoch(self, **_):
        # Check if pretraining time is up
        stopped_pretraining_this_epoch = False
        if self._is_pretraining:
            # If the following is false once, we never enter this branch again
            self._is_pretraining = \
                not self._duration.match(iteration_count=self.trainer.iteration_count,
                                         epoch_count=self.trainer.epoch_count)
            stopped_pretraining_this_epoch = not self._is_pretraining

        if stopped_pretraining_this_epoch:
            self.save_pretrained_model()
            self.trainer.print("Pretraining done. Building criterion and optimizer for training.")
            self.trainer.build_criterion(**self._training_criterion)
            self.trainer.build_optimizer(**self._training_optimizer)
