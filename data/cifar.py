import logging
import numpy as np
import tensorflow as tf
from utils.adapt_data import adapt_labels

RANDOM_SEED = 42
RNG = np.random.RandomState(42)

logger = logging.getLogger(__name__)

def get_train(label, centered=False):
    """Get training dataset for CIFAR"""
    return _get_adapted_dataset("train", label, centered=centered)

def get_test(label, centered=False):
    """Get testing dataset for CIFAR"""
    return _get_adapted_dataset("test", label, centered=centered)

def get_shape_input():
    """Get shape of the dataset for CIFAR"""
    return (None, 32, 32, 3)

def get_shape_input_flatten():
    """Get shape of the flatten dataset for CIFAR"""
    return (None, 3072)

def get_shape_label():
    """Get shape of the labels in CIFAR dataset"""
    return (None,)

def num_classes():
    """Get number of classes in CIFAR dataset"""
    return 10

def _get_adapted_dataset(split, label=None, centered=False, flatten=False):
    """Gets the adapted dataset for the experiments

    Args :
            split (str): train, valid or test
            label (int): int in range 0 to 10, is the class/digit
                         which is considered outlier
            centered (bool): (Default=False) data centered to [-1, 1]
            flatten (bool): (Default=False) flatten the data
    Returns :
            (tuple): <training, testing> images and labels
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_labels,test_labels = train_labels[:,0],test_labels[:,0] #because cifar labels are weird

    indicies = np.argwhere(train_labels == int(label))
    mask_train  = np.invert(train_labels == int(label))

    train_images = train_images[mask_train]
    train_labels = train_labels[mask_train]

    if split == 'train':
        return train_images,train_labels


    elif split == 'test':
        return test_images, test_labels == int(label)
