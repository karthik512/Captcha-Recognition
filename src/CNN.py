import tensorflow as tf
from src.Constants import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding='SAME')


def convolutional_layers():
    x = tf.placeholder(tf.float32, [None, None])

    # First layer
    W_conv1 = weight_variable([5, 5, 3, 24])
    b_conv1 = bias_variable([24])
    x_expanded = tf.reshape(x, [-1, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH])
    h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # Second layer
    W_conv2 = weight_variable([5, 5, 24, 36])
    b_conv2 = bias_variable([36])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(4, 2), stride=(4, 2))

    return x, h_pool2, [W_conv1, b_conv1, W_conv2, b_conv2]


def get_network():
    x, conv_layer, conv_vars = convolutional_layers()

    # Densely connected layer
    W_fc1 = weight_variable([20 * 15 * 36, 4096])
    b_fc1 = bias_variable([4096])

    conv_layer_flat = tf.reshape(conv_layer, [-1, 20 * 15 * 36])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)

    # Output layer
    W_fc2 = weight_variable([4096, TOTAL_CHARS * CLASSES])
    b_fc2 = bias_variable([TOTAL_CHARS * CLASSES])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    return (x, y, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])