#################################
# Your name: Chen Dar
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
# from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = 0, 8
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the final classifier
    """
    validate_data_and_labels(data=data, labels=labels)
    w = np.zeros(shape=get_x_dim(data), dtype=np.float128)
    # np.seterr(all='raise')
    for t in range(1, T + 1):
        random_choice = np.random.randint(len(data))
        y = labels[random_choice]
        x = data[random_choice]

        w = update_w(C, eta_0, t, w, x, y)

    return w
    # convert back from float128 to float64
    # final_w = np.ndarray(shape=get_x_dim(data), dtype=np.float64)
    # for i, wi in enumerate(w):
    #     final_w[i] = wi
    # return final_w


def update_w(C, eta_0, t, w, x, y):
  eta_t = eta_0 / t
  if np.inner(x, w) * y < 1:
    w = (1 - eta_t) * w + (eta_t * C * y) * x
  else:
    w = (1 - eta_t) * w
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


def sgd_predict(x, w):
    if np.inner(x, w) >= 0:
        return 1
    return -1

#################################
