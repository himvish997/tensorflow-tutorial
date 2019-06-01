# LeNet-5 Model
# filename: initializable_iterator.py

import tensorflow as tf
from model import model

# Initializable iterator
'''You can dynamically change the Dataset between training and validation 
Datasets. However, in this case both the Datasets needs to go through the 
same transformation pipeline.'''
def initializable_iterator(X_train, y_train, X_val, y_val):
    epochs = 10
    batch_size = 64
    tf.reset_default_graph()
    X_data = tf.placeholder(tf.float32, [None, 32,32,1])
    Y_data = tf.placeholder(tf.float32, [None, 10])
    dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    X_batch , Y_batch = iterator.get_next()
    (learner, accuracy) = model(X_batch, Y_batch)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):

            # train the model
            sess.run(iterator.initializer, feed_dict={X_data:X_train, Y_data:y_train})
            total_train_accuracy = 0
            no_train_examples = len(y_train)
            try:
                while True:
                    temp_train_accuracy, _ = sess.run([accuracy, learner])
                    total_train_accuracy += temp_train_accuracy*batch_size
            except tf.errors.OutOfRangeError:
                pass

            # validate the model
            sess.run(iterator.initializer, feed_dict={X_data:X_val, Y_data:y_val})
            total_val_accuracy = 0
            no_val_examples = len(y_val)
            try:
                while True:
                    temp_val_accuracy = sess.run(accuracy)
                    total_val_accuracy += temp_val_accuracy*batch_size
            except tf.errors.OutOfRangeError:
                pass

            print('Epoch {}'.format(str(epoch+1)))
            print("---------------------------")
            print('Training accuracy is {}'.format(total_train_accuracy/no_train_examples))
            print('Validation accuracy is {}'.format(total_val_accuracy/no_val_examples))