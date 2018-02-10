import numpy as np
import importlib

def adapt_labels(true_labels, label):
    """Adapt labels to anomaly detection context

    Args :
            true_labels (list): list of ints
            label (int): label which is considered anomalous
    Returns :
            true_labels (list): list of labels, 1 for anomalous and 0 for normal
    """
    if label == 0:
        (true_labels[true_labels == label], true_labels[true_labels != label]) = (0, 1)
        true_labels = [1] * true_labels.shape[0] - true_labels
    else:
        (true_labels[true_labels != label], true_labels[true_labels == label]) = (0, 1)

    return true_labels



def generate_batch_data_and_labels(dataset, label, number_ini_z, number_batchs, allow_smaller_batch=False):
    """ FOR GAN INVERSION ONLY

    Generate batch of data with same image number_ini_z times (non convex space) for the
    generator when inverting the gan

    Args :
            dataset (str): name of the dataset, mnist or cifar10
            label (int): label which is considered anomalous
            number_ini_z (int): number of z we want for parallel computation
            number_batchs (int) number of batches dividing the total testing set
    Returns :
            splits (list): list of np.arrays [number_ini_z * batch_number, -1]
            labels_test (list): list of labels, 1 for anomalous and 0 for normal
    """
    data = importlib.import_module('data.{}'.format(dataset))
    x_test, y_test = data.get_test(label, True)
    x_number = x_test.shape[0]
    two_splits = np.split(x_test, [x_number - x_number % number_batchs])
    repeat_first_split = np.repeat(two_splits[0], number_ini_z, axis=0)
    first_splits = np.split(repeat_first_split, number_batchs)
    second_splits = np.split(np.repeat(two_splits[1], number_ini_z, axis=0), 1)

    if two_splits[1].shape[0] != 0 and allow_smaller_batch:
        return first_splits+second_splits, y_test
    return first_splits, y_test[:two_splits[0].shape[0]]



