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

# --- Basic Setup --- #


# set up the data
data = dataHandler.DataHandler('data/imu_output.txt', 'data/ground_truth.txt', batch_size)
print(data.next_batch())

# create the network
network = networkStructure.network(input_size, time_steps)



estimate = tf.placeholder(tf.float64, shape=[None, 7])
