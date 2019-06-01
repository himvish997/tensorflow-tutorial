# LeNet-5 Model

import tensorflow as tf

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
