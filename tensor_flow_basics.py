import tensorflow as tf
import numpy as np

# first, create a TensorFlow constant
const = tf.constant(2.0, name="const")

# create TensorFlow variables
b = tf.Variable(2.0, name='b')
c = tf.Variable(1.0, name='c')



# now create some operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# setup the variable initialisation
init_op = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a)
    print("Variable a is {}".format(a_out))

b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, name='c')
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))
