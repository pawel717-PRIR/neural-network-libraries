from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras.datasets import mnist
from keras import backend as K
import keras
from keras.datasets import cifar10
from keras.datasets import cifar100
import pandas
import numpy as np
from sklearn.model_selection import train_test_split



def LoadMnistData():
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_data = np.concatenate((x_train, x_test))
    y_data = np.concatenate((y_train, y_test))
    y_data = keras.utils.to_categorical(y_data, num_classes)

    return x_data, y_data, input_shape, num_classes

def LoadCifar10Data():
    num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_data = np.concatenate((x_train, x_test))
    y_data = np.concatenate((y_train, y_test))
    y_data = keras.utils.to_categorical(y_data, num_classes)

    return x_data, y_data, num_classes

def LoadCifar100Data():
    num_classes = 100

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    x_data = np.concatenate((x_train, x_test))
    y_data = np.concatenate((y_train, y_test))
    y_data = keras.utils.to_categorical(y_data, num_classes)

    return x_data, y_data, num_classes


def LoadLetterRecognitionData():
    num_classes = 26
    df = pandas.read_csv("letter-recognition.csv", header=None)
    y_data = df[0]
    x_data = df.drop(df.columns[0], axis=1)

    y_data = pandas.factorize(y_data)
    return x_data.values, y_data[0], num_classes

def SplitData(x_data, y_data, num_classes):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test

def SplitDataForLetterRecognition(x_data, y_data, num_classes):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    y_train = pandas.factorize(y_train)
    y_test = pandas.factorize(y_test)

    y_train = keras.utils.to_categorical(y_train[0], num_classes)
    y_test = keras.utils.to_categorical(y_test[0], num_classes)

    return x_train.values, x_test.values, y_train, y_test


