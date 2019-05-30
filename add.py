import tensorflow as tf

x = tf.constant([3, 4, 5], name = 'x')
y = tf.constant([1, 2, 3], name = 'y')

z = tf.add(x, y, name='z')

with tf.Session() as sess:
    with tf.summary.FileWriter('summaries', sess.graph) as writer:
        numpy_z = sess.run([z])
print(numpy_z)
