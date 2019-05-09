
import tensorflow as tf
import numpy as np

def Print(next_ele):
    with tf.Session() as sess:
        try:
            while True:
                val = sess.run(next_ele)
                print(val)
        except tf.errors.OutOfRangeError:
            pass

###############################################################################
'''Batches: Combines consecutive elements of the Dataset into a single batch.
Useful when you want to train smaller batches of data to avoid out of memory 
errors.'''
###############################################################################

def Batches():
    data = np.arange(10, 40)

    # create batches of 10
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(10)

    # creat the iterator to consume the data
    iterator = dataset.make_one_shot_iterator()
    next_ele = iterator.get_next()
    return next_ele

'''
The output is :

[10 11 12 13 14 15 16 17 18 19] 
[20 21 22 23 24 25 26 27 28 29] 
[30 31 32 33 34 35 36 37 38 39]
'''


###############################################################################
'''Zip: Creates a Dataset by zipping together datasets. Useful in scenarios
where you have features and labels and you need to provide the pair of feature
and label for training the model.'''
###############################################################################

def Zip():
    data_x = np.arange(10, 40)
    data_y = np.arange(11, 41)

    dataset_x = tf.data.Dataset.from_tensor_slices(data_x)
    dataset_y = tf.data.Dataset.from_tensor_slices(data_y)

    dcombined = tf.data.Dataset.zip((dataset_x, dataset_y)).batch(5)

    iterator = dcombined.make_one_shot_iterator()
    return iterator.get_next()

'''
The output is

(array([10, 11]), array([11, 12])) 
(array([12, 13]), array([13, 14])) 
(array([14, 15]), array([15, 16])) 
(array([16, 17]), array([17, 18])) 
(array([18, 19]), array([19, 20]))
'''


###############################################################################
'''Repeat: Used to repeat the Dataset.'''
###############################################################################

def Repeat():
    dataset = tf.data.Dataset.from_tensor_slices(tf.range(10))
    dataset = dataset.repeat(count = 2)
    iterator = dataset.make_one_shot_iterator()
    next_ele = iterator.get_next()
    return next_ele

'''The Output is 

0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
'''


###############################################################################
'''Map: Used to transform the elements of the Dataset. Useful in cases where
you want to transform your raw data before feeding into the model.'''
###############################################################################

def map_fnc(x):
    return x*2
    

def Map():
    data = np.arange(10)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(map_fnc)

    iterator = dataset.make_one_shot_iterator()
    next_ele = iterator.get_next()
    return next_ele

'''
The output is

0 2 4 6 8 10 12 14 16 18
'''

###############################################################################
############# print the output of all the function ############################
###############################################################################

#Print(Batches())
#Print(Zip())
#Print(Repeat())
#Print(Map())



###############################################################################
##############  Iterators #####################################################
###############################################################################

def Print_One_Ite(next_element):
    with tf.Session() as sess:
        val = sess.run(next_element)
        print(val)


###############################################################################
'''One-shot iterator: This is the most basic form of iterator. It requires
no explicit initialization and iterates over the data only one time and once
it gets exhausted, it cannot be re-initialized.'''
###############################################################################

def One_Shot_iterator():
    data = np.arange(10, 50)
    #create the dataset
    dataset = tf.data.Dataset.from_tensor_slices(data)

    #Create the iterator
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element

# Print_One_Ite(One_Shot_iterator())


###############################################################################
'''Initializable iterator: This iterator requires you to explicitly initialize 
the iterator by running iterator.initialize. You can define a tf.placeholder 
and pass data to it dynamically each time you call the initialize 
operation.'''
###############################################################################

def Initializable_iterator(min_val = 10, max_val = 40, batch_size = 3):
    # Define two placeholders to accept min and max values
    _min_val = tf.placeholder(tf.int32, shape=[], name = 'min_val')
    _max_val = tf.placeholder(tf.int32, shape=[], name = 'max_val')
    _batch_size = tf.placeholder(tf.int64, shape=[], name = 'batch_size')

    data = tf.range(_min_val, _max_val)

    dataset = tf.data.Dataset.from_tensor_slices(data).batch(_batch_size)

    iterator = dataset.make_initializable_iterator()
    next_ele = iterator.get_next()
    with tf.Session() as sess:

        # Initialize an iterator with range of values from 10 to 16
        sess.run(
            iterator.initializer,
            feed_dict = {
                _min_val: min_val,
                _max_val: max_val,
                _batch_size: batch_size
                }
            )
        try:
            while True:
                val = sess.run(next_ele)
                print(val)
        except tf.errors.OutOfRangeError:
                pass
'''
The output is

[10 11 12]
[13 14 15]
[16 17 18]
[19 20 21]
[22 23 24]
[25 26 27]
[28 29 30]
[31 32 33]
[34 35 36]
[37 38 39]
'''


###############################################################################
'''Reinitializable iterator: This iterator can be initialized from different
Dataset objects that have the same structure. Each dataset can pass through 
its own transformation pipeline.'''
###############################################################################

def map_fnc(ele):
    return ele*2

def Reinitializable_Iterator(
        min_val_train = 10, 
        max_val_train = 18, 
        min_val_validation = 1, 
        max_val_validation = 10, 
        batch_size = 3
        ):
    min_val_ = tf.placeholder(tf.int32, shape = [], name = 'min_val')
    max_val_ = tf.placeholder(tf.int32, shape = [], name = 'max_val')
    batch_size_ = tf.placeholder(tf.int64, shape = [], name = 'batch_size')

    data = tf.range(min_val_, max_val_)

    # Define separate datasets for training and validation
    train_dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size_)
    val_dataset = tf.data.Dataset.from_tensor_slices(data).map(map_fnc).batch(batch_size_)

    # Create an iterator
    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes
        )

    train_initializer = iterator.make_initializer(train_dataset)
    val_initializer = iterator.make_initializer(val_dataset)

    next_ele = iterator.get_next()
    with tf.Session() as sess:
        print('Train Dataset:')
        # initialize an iterator with range of values from 10 to 16
        sess.run(train_initializer, feed_dict={
            min_val_:min_val_train,
            max_val_:max_val_train,
            batch_size_:batch_size
        })
        try:
            while True:
                val = sess.run(next_ele)
                print(val)
        except tf.errors.OutOfRangeError:
            pass

        print("Validation Dataset:")
        # Initialize an iterator with range of values from 1 to 10
        sess.run(val_initializer, feed_dict={
            min_val_:min_val_validation,
            max_val_:max_val_validation,
            batch_size_:batch_size
            })
        try:
            while True:
                val = sess.run(next_ele)
                print(val)
        except tf.errors.OutOfRangeError:
            pass
'''
The Output is:
Train Dataset:
[10 11 12]
[13 14 15]
[16 17]
Validation Dataset:
[2 4 6]
[ 8 10 12]
[14 16 18]
'''


