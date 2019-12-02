#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt

"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def perceptron(data, labels):
    """
	returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """

    validate_data_and_labels(data=data, labels=labels)
    w = np.zeros(shape=get_x_dim(data), dtype=float)

    for idx, x in enumerate(data):
        y = labels[idx]
        normalized_x = x / np.linalg.norm(x)
        sign = calc_sign(normalized_x, w)
        if sign == y:
            continue

        w = w + (y * normalized_x)

    return w


#################################

# Place for additional code

def validate_data_and_labels(data, labels):
    if not isinstance(data, np.ndarray) or not isinstance(labels, np.ndarray):
        raise Exception(f"data or labels are not of type numpy.ndarray")

    if data is None or data.size == 0 or labels is None or labels.size == 0:
        raise Exception(f"invalid data set or labels")

    if data.shape[0] > labels.shape[0]:
        raise Exception(f"not enough labels for data")

    if len(data.shape) < 2 or data.shape[1] == 0:
        raise Exception(f"invalid data")


def get_x_dim(data):
    return data.shape[1]

def calc_sign(x, w):
    return np.sign(np.inner(x, w))
#################################
