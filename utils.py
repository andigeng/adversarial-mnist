import os
from tensorflow.examples.tutorials.mnist import input_data


__all__ = [
    'CURR_DIR',
    'DATA_DIR',
    'SAVE_DIR',
    'get_data'
    ]


CURR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CURR_DIR, "MNIST_data")
SAVE_DIR = os.path.join(CURR_DIR, "saved_models/model.ckpt")


def get_data(directory=DATA_DIR):
    return input_data.read_data_sets(directory, one_hot=True)