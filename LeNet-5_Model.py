# LeNet-5 Model
'''Let’s import the MNIST data from the tensorflow library. The MNIST database
contains 60,000 training images and 10,000 testing images. Each image is of 
size 28*28*1. We need to resize it to 32*32*1 for the LeNet-5 Model.'''

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot = True)
X_train, y_train = mnist.train.images, mnist.train.labels
X_val, y_val = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels
X_train = np.pad(X_train, ((0,0), (2,2), (2,2), (0,0)), 'constant')
X_val =   np.pad(X_val, ((0,0), (2,2), (2,2), (0,0)), 'constant')
X_test =  np.pad(X_test, ((0,0), (2,2), (2,2), (0,0)), 'constant')


# Let’s define the forward propagation of the model.
def forward_pass(X):
    W1 = tf.get_variable("W1", [5,5,1,6], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    # for conv layer2
    W2 = tf.get_variable("W2", [5,5,6,16], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding='VALID')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding='VALID')
    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding='VALID')
    A2= tf.nn.relu(Z2)
    P2= tf.nn.max_pool(A2, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')
    P2 = tf.contrib.layers.flatten(P2)
   
    Z3 = tf.contrib.layers.fully_connected(P2, 120)
    Z4 = tf.contrib.layers.fully_connected(Z3, 84)
    Z5 = tf.contrib.layers.fully_connected(Z4,10, activation_fn= None)
    return Z5


# Let’s define the model operations
def model(X,Y):
    logits = forward_pass(X)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0009)
    learner = optimizer.minimize(cost)
    correct_predictions = tf.equal(tf.argmax(logits,1),   tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return (learner, accuracy)


# One-shot iterator
'''The Dataset can’t be reinitialized once exhausted. 
To train for more epochs, you would need to repeat the Dataset before feeding 
to the iterator. This will require huge memory if the size of the data is 
large. It also doesn’t provide any option to validate the model.'''
def One_Shot_iterator():
    epochs = 10
    batch_size = 64
    iterations = len(y_train) * epochs
    tf.reset_default_graph()
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # need to repeat the dataset for epoch number of times, as all the data needs
    # to be fed to the dataset at once
    dataset = dataset.repeat(epochs).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    X_batch, Y_batch = iterator.get_next()
    (learner, accuracy) = model(X_batch, Y_batch)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_accuracy = 0
        try:
            while True:
                temp_accuracy, _ = sess.run([accuracy, learner])
                total_accuracy += temp_accuracy
                print('Training accuracy is {}'.format((total_accuracy * batch_size) / iterations))

        except tf.errors.OutOfRangeError:
            pass

    print('Avg training accuracy is {}'.format((total_accuracy * batch_size) / iterations))


One_Shot_iterator()