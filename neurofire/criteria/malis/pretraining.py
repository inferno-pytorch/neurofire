from inferno.trainers.callbacks.base import Callback
from inferno.utils.train_utils import Frequency


class Pretrain(Callback):
    def __init__(self, duration=None):
        super(Pretrain, self).__init__()
        # Privates
        self._duration = None
        self._pretraining_criterion = {}
        self._training_criterion = {}
        self._pretraining_optimizer = {}
        self._training_optimizer = {}
        self._is_pretraining = True
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

    def set_training_criterion(self, method, **kwargs):
        self._training_criterion.update({'method': method})
        self._training_criterion.update(kwargs)

    def set_pretraining_optimizer(self, method, param_groups=None, **kwargs):
        self._pretraining_optimizer.update({'method': method, 'param_groups': param_groups})
        self._pretraining_optimizer.update(kwargs)

    def set_training_optimizer(self, method, param_groups=None, **kwargs):
        self._training_optimizer.update({'method': method, 'param_groups': param_groups})
        self._training_optimizer.update(kwargs)

    def begin_of_fit(self, **_):
        self.trainer.build_criterion(**self._pretraining_criterion)
        self.trainer.build_optimizer(**self._pretraining_optimizer)

    def begin_of_training_iteration(self, **_):
        # Check if pretraining time is up
        if self._is_pretraining:
            self._is_pretraining = \
                self._duration.match(iteration_count=self.trainer.iteration_count,
                                     epoch_count=self.trainer.epoch_count)
        pass

    def begin_of_epoch(self):
        pass
