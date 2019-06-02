# LeNet-5 Model
# filename: main.py

import tensorflow as tf
import numpy as np
import argparse
from One_shot_iterator import One_Shot_iterator
from Initializable_iterator import initializable_iterator
from ReInitializable_iterator import ReInitializable_iterator
from Feedable_iterator import Feedable_iterator
from download_data import train_data, val_data, test_data

parser = argparse.ArgumentParser(description='Training the LeNet-5 model with the variours data pipeline method (Iterator) in the tensorflow')
parser.add_argument("-osi", "--one_shot_iterator", help="One-shot iterator: The Dataset can’t be reinitialized once exhausted. To train for more epochs, you would need to repeat the Dataset before feeding to the iterator. This will require huge memory if the size of the data is large. It also doesn’t provide any option to validate the model.", action='store_true')
parser.add_argument("-ii", "--initializable_iterator", help="Initializable iterator: You can dynamically change the Dataset between training and validation Datasets. However, in this case both the Datasets needs to go through the same transformation pipeline.",
                    action='store_true')
parser.add_argument("-ri", "--reinitializable_iterator", help="Re-initializable iterator: This iterator overcomes the problem of initializable iterator by using two separate Datasets. Each dataset can go through its own preprocessing pipeline. The iterator can be created using the tf.Iterator.from_structure method.", action='store_true')
parser.add_argument("-fi", "--feedable_iterator", help="Feedable iterator: This iterator provides the option of switching between various iterators. You can create a re-initializable iterator for training and validation purposes. For inference/testing where you require one pass of the dataset, you can use the one shot iterator.", action='store_true')
args = parser.parse_args()

def main():
    print("LeNet-5 model runs with various Iterators that TensorFlow provides")

    if args.one_shot_iterator:
        X_train, Y_train = train_data()
        One_Shot_iterator(X_train, Y_train)
    elif args.initializable_iterator:
        X_train, Y_train = train_data()
        X_val, Y_val = val_data()
        initializable_iterator(X_train, Y_train, X_val, Y_val)
    elif args.reinitializable_iterator:
        X_train, Y_train = train_data()
        X_val, Y_val = val_data()
        ReInitializable_iterator(X_train, Y_train, X_val, Y_val)
    elif args.feedable_iterator:
        X_train, Y_train = train_data()
        X_val, Y_val = val_data()
        X_test, Y_test = test_data()
        Feedable_iterator(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    else:
        print("Please give some argparse. For more try:\n    python3 main.py -h or --help")


if __name__ == '__main__':
    main()
