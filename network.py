# TensorFlow
import tensorflow as tf

# Network
import networkStructure
# DataHandler
import dataHandler


# ########### #
# Basic Setup #
# ########### #

# sensor input
input_size  = 67
# output classes (7 degrees of freedom)
num_classes = 7
# process one at a time with the LSTM
time_steps = 1
# batch size
batch_size = 15

# training epochs
epochs = 1
learning_rate = 0.01

# --- Basic Setup --- #


# set up the data
data = dataHandler.DataHandler('data/imu_output.txt', 'data/ground_truth.txt', batch_size)


# create the network
network = networkStructure.network(input_size, time_steps)

# estimation
estimate = tf.placeholder(tf.float64, shape=[None, 7])

# cost function and optimization
cost = tf.reduce_sum(tf.pow(estimate - network, 2)) / (2 * data.training_data_size())
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# initialize session
init = tf.global_variables_initializer()

# training loop
with tf.Session() as sess:
    sess.run(init)
