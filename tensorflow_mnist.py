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


leaky_relu_alpha = 0.2
dropout_rate = 0.5
padding = "SAME"


def conv2d(inputs, filters, biases, stride_size):
    out = tf.nn.conv2d(inputs, filters, strides=[1, stride_size, stride_size, 1], padding=padding)
    out = tf.nn.bias_add(out, biases)
    return tf.nn.relu(out)


def maxpool(inputs, pool_size, stride_size):
    return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], padding='VALID',
                            strides=[1, stride_size, stride_size, 1])


def dense(inputs, weights, dropout_rate):
    x = tf.nn.relu(tf.matmul(inputs, weights))
    return tf.nn.dropout(x, rate=dropout_rate)


batch_size = 128


def create_dataset(xs, ys, n_classes=10):
    ys = tf.one_hot(ys, depth=n_classes)
    return tf.data.Dataset.from_tensor_slices((xs, ys)) \
        .map(preprocess) \
        .shuffle(len(ys)) \
        .batch(batch_size)


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

num_classes = np.unique(y_train).size
# preprocess
# x_train, y_train = preprocess(x_train, y_train)
# x_test, y_test = preprocess(x_test, y_test)
dataset = create_dataset(x_train, y_train, num_classes)

# y_train = tf.one_hot(y_train, depth=num_classes)
# y_test = tf.one_hot(y_test, depth=num_classes)

initializer = tf.initializers.glorot_uniform()


def get_weight(shape, name):
    return tf.Variable(initializer(shape), name=name, trainable=True, dtype=tf.float32)


def model(x, weights, biases):
    conv1 = conv2d(x, weights[0], biases['bc1'], stride_size=1)
    conv2 = conv2d(conv1, weights[1], biases['bc2'], stride_size=1)
    maxpool1 = maxpool(conv2, pool_size=2, stride_size=2)
    drp1 = tf.nn.dropout(maxpool1, 0.25)

    flatten = tf.reshape(maxpool1, shape=(tf.shape(drp1)[0], -1))
    dns = dense(flatten, weights[2], dropout_rate=0.5)

    aa = tf.matmul(dns, weights[3])

    logits = aa
    prediction = tf.nn.softmax(logits)

    return prediction


def loss(pred, target):
    return tf.losses.categorical_crossentropy(target, pred)


def train_step(model, inputs, outputs):
    with tf.GradientTape() as tape:
        current_loss = loss(model(inputs, weights, biases), outputs)
    grads = tape.gradient(current_loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    return tf.reduce_mean(current_loss)
    # print( tf.reduce_mean( current_loss ))


# shape [kernel_height, kernel_width, in_channels, num_filters]
shapes = [
    [3, 3, 1, 32],
    [3, 3, 32, 64],
    [12544, 64],
    [64, num_classes],
]

biases = {
    'bc1': tf.Variable(tf.random.normal([32]), trainable=True),
    'bc2': tf.Variable(tf.random.normal([64]), trainable=True),
    'out': tf.Variable(tf.random.normal([num_classes]), trainable=True)
}

weights = []
for i in range(len(shapes)):
    weights.append(get_weight(shapes[i], 'weight{}'.format(i)))

optimizer = tf.keras.optimizers.Adadelta()

# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=y_train))
# train_op = optimizer.minimize(loss_op)
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_train, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

num_epochs = 12
cur = 0
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    for features in dataset:
        image, label = features[0], features[1]
        cur = train_step(model, image, label)
    print("epoch: ", epoch, "cur: ", cur)
