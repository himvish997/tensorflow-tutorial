# LeNet-5 Model
# filename: One-shot-iterator.py

import tensorflow as tf
from model import model

# One-shot-iterator
'''The Dataset can’t be reinitialized once exhausted. 
To train for more epochs, you would need to repeat the Dataset before feeding 
to the iterator. This will require huge memory if the size of the data is 
large. It also doesn’t provide any option to validate the model.'''
def One_Shot_iterator(X_train, Y_train):
    print("One Shot Iterator")
    epochs = 10
    batch_size = 64
    iterations = len(Y_train) * epochs
    tf.reset_default_graph()
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
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


