from .base import Server


class SerialDataParallelServer(Server):
    def __init__(self, model=None):
        super(SerialDataParallelServer, self).__init__(model=model)
        # Privates
        self._devices = None
        self._max_batchsize_per_device = None

    def cuda(self):
        # TODO
        return self


