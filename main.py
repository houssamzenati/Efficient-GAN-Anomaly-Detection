#! python3

import argparse
import importlib
import logging
import os
import shutil
import urllib3
import zipfile

import data

# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logger = logging.getLogger("AnomalyDetection")


def run(args):
    print("""
 ______   _____       _____       ____   
|_     `.|_   _|     / ___ `.   .'    '. 
  | | `. \ | |      |_/___) |  |  .--.  |
  | |  | | | |   _   .'____.'  | |    | |
 _| |_.' /_| |__/ | / /_____  _|  `--'  |
|______.'|________| |_______|(_)'.____.' 
                                         
""")

    has_effect = False

    if args.example and args.dataset and args.split:
        try:
            mod_name = "{}.{}_{}".format(args.example, args.split, args.dataset)
            logger.info("Running script at {}".format(mod_name))

            mod = importlib.import_module(mod_name)
            mod.run(args.nb_epochs, args.w, args.m, args.d, args.label, args.rd)

        except Exception as e:
            logger.exception(e)
            logger.error("Uhoh, the script halted with an error.")
    else:
        if not has_effect:
            logger.error("Script halted without any effect. To run code, use command:\npython3 main.py <example name> {train, test, run}")

def path(d):
    try:
        assert os.path.isdir(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run examples from the DL 2.0 Anomaly Detector.')
    parser.add_argument('example', nargs="?", type=path, help='the folder name of the example you want to run e.g gan or bigan')
    parser.add_argument('dataset', nargs="?", choices=['mnist', 'kdd'], help='the name of the dataset you want to run the experiments on')
    parser.add_argument('split', nargs="?", choices=['run'], help='train the example or evaluate it')
    parser.add_argument('--nb_epochs', nargs="?", type=int, help='number of epochs you want to train the dataset on')
    parser.add_argument('--label', nargs="?", type=int, help='anomalous label for the experiment')
    parser.add_argument('--w', nargs="?", default=0.1, type=float, help='weight for the sum of the mapping loss function')
    parser.add_argument('--m', nargs="?", default='fm', choices=['cross-e', 'fm'], help='mode/method for discriminator loss')
    parser.add_argument('--d', nargs="?", default=1, type=int, help='degree for the L norm')
    parser.add_argument('--rd', nargs="?", default=42, type=int, help='random_seed')

    run(parser.parse_args())
