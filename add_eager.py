import tensorflow as tf
# import tf.eager
from tensorflow.contrib.eager.python import tfe

tfe.enable_eager_execution()

x = tf.constant([3, 4, 5])
y = tf.constant([1, 2, 3])

print("X - Y: ", x-y)
