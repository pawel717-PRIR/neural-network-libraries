import tensorflow as tf
import numpy as np


def uses_gpu():
    """
    Check if gpu is used by tensorflow library in computations
    :return: True if gpu available, False otherwise
    """
    print("Is built with cuda: ", tf.test.is_built_with_cuda())
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        return True
    else:
        print("Please install GPU version of TF")
        return False


def preprocess(x, y):
    """
    Preprocess data - scale to range [0, 1], cast x to float32, cast y to uint8
    :param x:
    :param y:
    :return: x, y given as input after preprocessing
    """
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.uint8)

    return x, y


def create_dataset(xs, ys, n_classes=10):
    ys = tf.one_hot(ys, depth=n_classes)
    return tf.data.Dataset.from_tensor_slices((xs, ys)) \
        .map(preprocess) \
        .shuffle(len(ys)) \
        .batch(128)


class MnistModel(tf.keras.Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(64, input_shape=(2,), activation='sigmoid')
        self.hidden2 = tf.keras.layers.Dense(128, activation='relu')
        self.out = tf.keras.layers.Dense(2)
        self.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MSE)

    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.out(x)
        return x


def FitMnistModel(model, x_train, y_train, x_test, y_test):
    batch_size = 128
    epochs = 12

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# input image dimensions
img_rows, img_cols = 28, 28

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# preprocess
x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)

num_classes = np.unique(y_train).size

y_train = tf.one_hot(y_train, depth=num_classes)
y_test = tf.one_hot(y_test, depth=num_classes)



model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                 activation='relu',
                                 input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

FitMnistModel(model, x_train, y_train, x_test, y_test)
