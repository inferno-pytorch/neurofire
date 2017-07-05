from neurofire.datasets.cremi.pipelines import basic
import argparse


CONFIG_FILE = '/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/' \
              'CREMI/CANTOR-MEMBRANES-1DS-3SLICE-0/Configurations/trainer_config.yml'


if __name__ == '__main__':
    if CONFIG_FILE is None:
        # Parse arguments from commandline
        parsey = argparse.ArgumentParser()
        parsey.add_argument('-config_file', help="Path to the config file.")
        args = parsey.parse_args()
    else:
        args = argparse.Namespace(config_file=CONFIG_FILE)
    # Build pipeline and train
    pipeline = basic.BasicPipeline.build(args.config_file)
    pipeline.train()

