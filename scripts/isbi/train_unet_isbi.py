from neuronfire.datasets.isbi2012.loaders import get_isbi_loader
from neuronfire.models import UNet2D

def train(use_gpu=False):
    # TODO affinities
    model = UNet2D(1, 2) # 1 grayscale input channel, 1 output channel (membrane probability)

    train_loader, validate_loader = get_isbi_loader('./data_config.yml')

    trainer = Trainer(model)  # TODO all the params / settings

    trainer.bind_loader('train', train_loader)
    trainer.bind_loader('test', test_loader)

    if use_gpu:
        trainer.cuda()

    trainer.fit()


if __name__ == '__main__':
    train()
