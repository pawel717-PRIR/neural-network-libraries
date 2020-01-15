from __future__ import print_function
import tensorflow as tf


class MnistModel(tf.keras.Model):
    def __init__(self, num_classes, input_shape):
        super(MnistModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                            input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.flatten1 = tf.keras.layers.Flatten()
        self.droput1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.droput1(x)
        x = self.dense2(x)
        return x


class Cifar10Model(tf.keras.Model):
    def __init__(self, num_classes, input_shape):
        super(Cifar10Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                            padding='same', input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.droput3 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.droput3(x)
        x = self.dense2(x)
        return x


class Cifar100Model(tf.keras.Model):
    def __init__(self, num_classes, input_shape):
        super(Cifar100Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='elu',
                                            padding='same', input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)

        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='elu')
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='elu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        return x

def CreateMnistModel(input_shape, num_classes):
    return MnistModel(input_shape=input_shape, num_classes=num_classes)


def CreateCifar10Model(num_classes, x_train):
    return Cifar10Model(input_shape=x_train.shape[1:], num_classes=num_classes)


def CreateCifar100Model(num_classes, x_train):
    return Cifar100Model(input_shape=x_train.shape[1:], num_classes=num_classes)
