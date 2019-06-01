# LeNet-5 Model
# filename: Re-initializable_iterator.py

import tensorflow as tf
from model import model

# Re-initializable iterator
'''This iterator overcomes the problem of initializable iterator by using two 
separate Datasets. Each dataset can go through its own preprocessing pipeline.
The iterator can be created using the tf.Iterator.from_structure method.'''


def map_fnc(X, Y):
    return X, Y

def ReInitializable_iterator(X_train, y_train, X_val, y_val):
    epochs = 10
    batch_size = 64
    tf.reset_default_graph()
    X_data = tf.placeholder(tf.float32, [None, 32, 32, 1])
    Y_data = tf.placeholder(tf.float32, [None, 10])
    train_dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data)).batch(batch_size).map(map_fnc)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data)).batch(batch_size)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    X_batch, Y_batch = iterator.get_next()
    (learner, accuracy) = model(X_batch, Y_batch)
    train_initializer = iterator.make_initializer(train_dataset)
    val_initializer = iterator.make_initializer(val_dataset)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):

            # train the model
            sess.run(train_initializer, feed_dict={X_data: X_train, Y_data: y_train})
            total_train_accuracy = 0
            no_train_examples = len(y_train)
            try:
                while True:
                    temp_train_accuracy, _ = sess.run([accuracy, learner])
                    total_train_accuracy += temp_train_accuracy * batch_size
            except tf.errors.OutOfRangeError:
                pass

            # validate the model
            sess.run(val_initializer, feed_dict={X_data: X_val, Y_data: y_val})
            total_val_accuracy = 0
            no_val_examples = len(y_val)
            try:
                while True:
                    temp_val_accuracy = sess.run(accuracy)
                    total_val_accuracy += temp_val_accuracy * batch_size
            except tf.errors.OutOfRangeError:
                pass

            print('Epoch {}'.format(str(epoch + 1)))
            print("---------------------------")
            print('Training accuracy is {}'.format(total_train_accuracy / no_train_examples))
            print('Validation accuracy is {}'.format(total_val_accuracy / no_val_examples))
