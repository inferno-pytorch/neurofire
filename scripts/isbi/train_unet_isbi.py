from __future__ import print_function

import os

from neurofire.datasets.isbi2012.loaders import get_isbi_loader
from neurofire.models import UNet2D

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

PROJECT_DIRECTORY = '.'


# TODO affinities + malis
def train(gpu_list=[]):

    # 1 grayscale input channel, 1 output channel (membrane probability)
    model = UNet2D(1, 1, n_scale=2)

    train_loader = get_isbi_loader('./data_config.yml')

    trainer = Trainer(model)
    trainer.build_criterion('SorensenDiceLoss')
    trainer.build_optimizer('Adam')

    # TODO need validation
    # trainer.build_metric('')
    # trainer.validate_every((1, 'epochs'))

    trainer.save_every(
        (1000, 'iterations'),
        to_directory=os.path.join(PROJECT_DIRECTORY, 'weights')
    )
    trainer.set_max_num_iterations(int(1e4))

    # Tensorboard logger
    trainer.build_logger(TensorboardLogger(
        send_image_at_batch_indices=0,
        send_image_at_channel_indices='all',
        log_images_every=(20, 'iterations')),
        log_directory=os.path.join(PROJECT_DIRECTORY, 'logs')
    )

    trainer.bind_loader('train', train_loader)
    # trainer.bind_loader('test', validate_loader)

    if gpu_list:
        trainer.cuda(gpu_list)

    trainer.fit()


if __name__ == '__main__':
    # gpu_list = [0]
    train()
