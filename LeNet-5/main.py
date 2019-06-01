# LeNet-5 Model
# filename: main.py

import tensorflow as tf
import numpy as np
import argparse
from One_shot_iterator import One_Shot_iterator
from Initializable_iterator import initializable_iterator
from download_data import train_data, val_data, test_data


parser = argparse.ArgumentParser(description='Running the LeNet-5 model with the variours Iterator in the tensorflow')
parser.add_argument("-osi", "--one_shot_iterator", help="Running by the One Shot Iterator", action='store_true')
parser.add_argument("-ii", "--initializable_iterator", help="Running by the initializable iterator",
                    action='store_true')
args = parser.parse_args()

def main():
    print("LeNet-5 model runs with various Iterators that TensorFlow provides")

    if args.one_shot_iterator:
        X_train, Y_train = train_data()
        One_Shot_iterator(X_train, Y_train)
    if args.initializable_iterator:
        X_train, Y_train = train_data()
        X_val, Y_val = val_data()
        initializable_iterator(X_train, Y_train, X_val, Y_val)





if __name__ == '__main__':
    main()