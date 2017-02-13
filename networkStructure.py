import tensorflow as tf

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

def network(input_size, time_steps):

    # Data Input Layer
    # 67 neurons
    dataLayer = tf.placeholder(tf.float64, shape=[None, time_steps, input_size])

    # Permuting batch_size and n_steps
    dataLayer = tf.transpose(dataLayer, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    dataLayer = tf.reshape(dataLayer, [-1, input_size])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    dataLayer = tf.split(0, time_steps, dataLayer)

    # #### LSTM ####

    # LSTM Cell
    lstm_cell  = tf.nn.rnn_cell.LSTMCell(256)
    output, _ = tf.nn.rnn(lstm_cell, dataLayer, dtype=tf.float64)

    lstm_weights = tf.Variable(tf.zeros([256, 128], dtype=tf.float64), dtype=tf.float64)
    lstm_bias    = tf.Variable(tf.zeros([128], dtype=tf.float64), dtype=tf.float64)



    # Fully Connected Layer
    # 128 neurons
    fc1 = tf.tanh(tf.matmul(output[-1], lstm_weights) + lstm_bias)
    fc1_weights = tf.Variable(tf.zeros([128, 128], dtype=tf.float64), dtype=tf.float64)
    fc1_bias    = tf.Variable(tf.zeros([128], dtype=tf.float64), dtype=tf.float64)




    # Fully Connected Layer
    # 128  neurons

    fc2 = tf.tanh(tf.matmul(fc1, fc1_weights) + fc1_bias)
    fc2_weights = tf.Variable(tf.zeros([128, 64], dtype=tf.float64), dtype=tf.float64)
    fc2_bias    = tf.Variable(tf.zeros([64], dtype=tf.float64), dtype=tf.float64)



    # Filly Connected Layer
    # 64 neurons

    fc3 = tf.tanh(tf.matmul(fc2, fc2_weights) + fc2_bias)
    fc3_weights = tf.Variable(tf.zeros([64, 7], dtype=tf.float64), dtype=tf.float64)
    fc3_bias    = tf.Variable(tf.zeros([7], dtype=tf.float64), dtype=tf.float64)



    # Pose Estimate
    # 7 neurons

    output = tf.matmul(fc3, fc3_weights) + fc3_bias

    return output
