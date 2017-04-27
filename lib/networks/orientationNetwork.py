import tensorflow as tf
import numpy as np

# Builds the network consisting of
#   Input layer             67
#
#   LSTM                   256
#
#   Fully Connected Layer  128
#   Fully Connected Layer  128
#   Fully Connected Layer   68
#
#   Output Layer             7

def Network(input_tensor):

    input_size = int(input_tensor.get_shape()[1])

    # Data Input Layer
    # 67 neurons

    # Has to be a list / sequence for the LSTM
    dataLayer = [input_tensor]


    # #### LSTM ####

    # LSTM Cell
    lstm_cell  = tf.nn.rnn_cell.LSTMCell(256, state_is_tuple=True)
    output, _ = tf.nn.rnn(lstm_cell, dataLayer, dtype=tf.float64)

    lstm_weights = tf.Variable(tf.random_normal([256, 128], dtype=tf.float64) / np.sqrt(256), dtype=tf.float64)
    lstm_bias    = tf.Variable(tf.zeros([128], dtype=tf.float64), dtype=tf.float64)



    # Fully Connected Layer
    # 128 neurons
    fc1 = tf.tanh(tf.matmul(output[-1], lstm_weights) + lstm_bias)
    fc1_weights = tf.Variable(tf.random_normal([128, 128], dtype=tf.float64) / np.sqrt(128), dtype=tf.float64)
    fc1_bias    = tf.Variable(tf.zeros([128], dtype=tf.float64), dtype=tf.float64)




    # Fully Connected Layer
    # 128  neurons

    fc2 = tf.tanh(tf.matmul(fc1, fc1_weights) + fc1_bias)
    fc2_weights = tf.Variable(tf.random_normal([128, 64], dtype=tf.float64) / np.sqrt(128), dtype=tf.float64)
    fc2_bias    = tf.Variable(tf.zeros([64], dtype=tf.float64), dtype=tf.float64)



    # Filly Connected Layer
    # 64 neurons

    fc3 = tf.tanh(tf.matmul(fc2, fc2_weights) + fc2_bias)
    fc3_weights = tf.Variable(tf.random_normal([64, 4], dtype=tf.float64) / np.sqrt(64), dtype=tf.float64)
    fc3_bias    = tf.Variable(tf.zeros([4], dtype=tf.float64), dtype=tf.float64)



    # Pose Estimate
    # 7 neurons

    output = tf.matmul(fc3, fc3_weights) + fc3_bias

    return output
