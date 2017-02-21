# TensorFlow
import tensorflow as tf

# DataHandler
import dataHandler
# Network
import networkStructure


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
batch_size = 1

# training epochs
epochs = 1
learning_rate = 0.01

# --- Basic Setup --- #


# ####### #
# Network #
# ####### #

# input
network_input = tf.placeholder(tf.float64, shape=[None, time_steps, input_size])

# output from the network described in networkStructure.py
output = networkStructure.network(network_input)


# estimation
estimate = tf.placeholder(tf.float64, shape=[None, 7])

# --- network --- #


# set up the data
data = dataHandler.DataHandler('data/imu_output.txt', 'data/ground_truth.txt', batch_size)



# cost function and optimization
cost = tf.reduce_sum(tf.pow(output - estimate, 2)) / (2 * data.training_data_size())
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# initialize session
init = tf.global_variables_initializer()

# training loop
with tf.Session() as sess:
    sess.run(init)

    i = 0

    while(data.data_available()):

        data, labels = data.next_batch()
        # print data
        # print labels
        sess.run(cost, {network_input: data, estimate: labels})
        # print("test")
        i = i+1
        if(i % 10 == 0):
            print ("Step %i" % i)
