import numpy as np
from tensorflow.keras.utils import to_categorical
from scipy.signal import convolve2d


def smooth_by_neighbours(one_hot_labels, kernel):
    kernel = np.array(kernel)
    assert kernel.ndim == 1
    return convolve2d(one_hot_labels, kernel.reshape(-1, 1),
                      mode="same", boundary='symm')


def smoothen(one_hot_labels, max_alpha, n_classes):
    alpha = np.random.rand() * max_alpha
    return (1-alpha) * one_hot_labels + (alpha/n_classes)
