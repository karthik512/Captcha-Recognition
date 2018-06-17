from PIL import Image
import src.CNN as network
import numpy as np
import os
import tensorflow as tf
from src.DataLoader import *
from src.Constants import *


def get_loss(y_pred, y_true):
    # Calculate the loss from digits being incorrect.  Don't count loss from
    # digits that are in non-present plates.
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                                          logits=tf.reshape(y_pred, [-1, CLASSES]),
                                          labels=tf.reshape(y_true, [-1, CLASSES]))
    loss = tf.reshape(loss, [-1, TOTAL_CHARS])
    loss = tf.reduce_sum(loss)
    return loss


dataDirectory = os.getcwd() + '\\..\\data'
dataLoader = DataLoader(dataDirectory)

training_data, validation_data, test_data = dataLoader.load_data()


mini_batchesX = [training_data[0][k:k + MINI_BATCH_SIZE] for k in range(0, len(training_data[0]), MINI_BATCH_SIZE)]
mini_batchesY = [training_data[1][k:k + MINI_BATCH_SIZE] for k in range(0, len(training_data[1]), MINI_BATCH_SIZE)]


x, y_pred, params = network.get_network()

y_true = tf.placeholder(tf.float32, [None, TOTAL_CHARS * CLASSES])

digits_loss = get_loss(y_pred, y_true)
train_step = tf.train.AdamOptimizer(1e-4).minimize(digits_loss)

predicted = tf.argmax(tf.reshape(y_pred, [-1, TOTAL_CHARS, CLASSES], name='Predicted'), 2)
correct = tf.argmax(tf.reshape(y_true, [-1, TOTAL_CHARS, CLASSES], name='Correct'), 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCHS):
        for idx, (X, Y) in enumerate(zip(mini_batchesX, mini_batchesY)):
            if ((idx + 1) % 50) == 0:
                result = sess.run([predicted, correct, digits_loss], feed_dict={x: X, y_true: Y})
                num_correct = np.sum(np.all(result[0] == result[1], axis=1))
                result_short = (result[0][:min(len(result[0]), 10)], result[1][:min(len(result[1]), 10)])
                for pred, corr in zip(*result_short):
                    print("{0}, {1} -> {2}".format(idx + 1, vec_to_captcha(pred), vec_to_captcha(corr)))
                print("%Correct :: {0} - Loss :: {1}".format(100. * num_correct / len(result[0]), result[2]))
            sess.run(train_step, feed_dict={x: X, y_true: Y})
        print(' -------------------------- Epoch {0} Completed -------------------------- '.format(epoch))

    mini_batchesX = [test_data[0][k:k + MINI_BATCH_SIZE] for k in range(0, len(test_data[0]), MINI_BATCH_SIZE)]
    mini_batchesY = [test_data[1][k:k + MINI_BATCH_SIZE] for k in range(0, len(test_data[1]), MINI_BATCH_SIZE)]
    total_test_num_correct = 0

    for idx, (X, Y) in enumerate(zip(mini_batchesX, mini_batchesY)):
        sess.run(train_step, feed_dict={x: X, y_true: Y})
        test_result = sess.run([predicted, correct, digits_loss], feed_dict={x: X, y_true: Y})
        test_num_correct = np.sum(np.all(test_result[0] == test_result[1], axis=1))
        total_test_num_correct += test_num_correct
        print("{0} - Test Data :: %Correct :: {1} - Loss :: {2}".format(idx, 100. * test_num_correct / len(test_result[0]), test_result[2]))
    print(" Test Data - Total Correct :: {0}".format(100. * total_test_num_correct / len(test_data[0])))
