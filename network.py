# TensorFlow
import tensorflow as tf

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

# --- Basic Setup --- #

# create the network
network = networkStructure.network(input_size, time_steps)



estimate = tf.placeholder(tf.float64, shape=[None, 7])
