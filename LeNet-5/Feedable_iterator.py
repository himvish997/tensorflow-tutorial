# LeNet-5 Model
# filename: Feedable_iterator.py

import tensorflow as tf
import numpy as np
from model import model

# Feedable iterator
'''This iterator provides the option of switching between various iterators. 
We can create a re-initializable iterator for training and validation purposes. 
For inference/testing where you require one pass of the dataset, We can use 
the one shot iterator.'''

def Feedable_iterator(X_train, y_train, X_val, y_val, X_test, y_test):
    epochs = 10
    batch_size = 64

    tf.reset_default_graph()

    X_data = tf.placeholder(tf.float32, [None, 32, 32, 1])
    Y_data = tf.placeholder(tf.float32, [None, 10])

    train_dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data)).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test.astype(np.float32))).batch(batch_size)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    X_batch, Y_batch = iterator.get_next()
    (learner, accuracy) = model(X_batch, Y_batch)

    train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_val_iterator.make_initializer(train_dataset)
    val_iterator = train_val_iterator.make_initializer(val_dataset)

    test_iterator = test_dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_val_string_handle = sess.run(train_val_iterator.string_handle())
        test_string_handle = sess.run(test_iterator.string_handle())
        for epoch in range(epochs):

            # train the model
            sess.run(train_iterator, feed_dict={X_data: X_train, Y_data: y_train})
            total_train_accuracy = 0
            no_train_examples = len(y_train)
            try:
                while True:
                    temp_train_accuracy, _ = sess.run([accuracy, learner], feed_dict={handle: train_val_string_handle})
                    total_train_accuracy += temp_train_accuracy * batch_size
            except tf.errors.OutOfRangeError:
                pass

            # validate the model
            sess.run(val_iterator, feed_dict={X_data: X_val, Y_data: y_val})
            total_val_accuracy = 0
            no_val_examples = len(y_val)
            try:
                while True:
                    temp_val_accuracy = sess.run(accuracy, feed_dict={handle: train_val_string_handle})
                    total_val_accuracy += temp_val_accuracy * batch_size
            except tf.errors.OutOfRangeError:
                pass

            print('Epoch {}'.format(str(epoch + 1)))
            print("---------------------------")
            print('Training accuracy is {}'.format(total_train_accuracy / no_train_examples))
            print('Validation accuracy is {}'.format(total_val_accuracy / no_val_examples))

        print("Testing the model --------")

        total_test_accuracy = 0
        no_test_examples = len(y_test)
        try:
            while True:
                temp_test_accuracy = sess.run(accuracy, feed_dict={handle: test_string_handle})
                total_test_accuracy += temp_test_accuracy * batch_size
        except tf.errors.OutOfRangeError:
            pass

        print('Testing accuracy is {}'.format(total_test_accuracy / no_test_examples))