# LeNet-5 Model
# filename: download_data.py
'''Let’s import the MNIST data from the tensorflow library. The MNIST database
contains 60,000 training images and 10,000 testing images. Each image is of
size 28*28*1. We need to resize it to 32*32*1 for the LeNet-5 Model.'''

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot = True)


def train_data():
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_train = np.pad(X_train, ((0,0), (2,2), (2,2), (0,0)), 'constant')
    return X_train, y_train


def val_data():
    X_val, y_val = mnist.validation.images, mnist.validation.labels
    X_val =   np.pad(X_val, ((0,0), (2,2), (2,2), (0,0)), 'constant')
    return X_val, y_val


def test_data():
    X_test, y_test = mnist.test.images, mnist.test.labels
    X_test =  np.pad(X_test, ((0,0), (2,2), (2,2), (0,0)), 'constant')
    return X_test, y_test