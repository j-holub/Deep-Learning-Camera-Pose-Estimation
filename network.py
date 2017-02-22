# TensorFlow
import tensorflow as tf

# Argument parser
import argparse
# filesystem stuf
import os
import sys

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
batch_size = 20

# training epochs
epochs = 100
# steps how often the intermediate result should be output
display_step = 10
# lerning rate for the gradient descent
learning_rate = 0.01

# --- Basic Setup --- #


# ################ #
# Argument Parsing #
# ################ #

parser = argparse.ArgumentParser()

parser.add_argument("IMU_Data", help="File holding the IMU measurements")
parser.add_argument("Ground_Truth", help="File holding the ground truth data")
parser.add_argument("--epochs", "-ep", default=100, help="Number of epochs to train")
parser.add_argument("--display_step", "-ds", default=10, help="How often intermediate results should be output")
parser.add_argument("--batch_size", "-bs", default=20, help="Batch Size for training")

arguments = parser.parse_args()

# check if the IMU file exists
if(not os.path.exists(arguments.IMU_Data)):
    print("'%s' not found" % arguments.IMU_Data)
    sys.exit()

# check if the Ground Truth file exists
if(not os.path.exists(arguments.Ground_Truth)):
    print("'%s' not found" % arguments.Ground_Truth)
    sys.exit()


# parse optional args
epochs = int(arguments.epochs)
display_step = int(arguments.display_step)
batch_size = int(arguments.batch_size)


# #### #
# Data #
# #### #

# set up the data
# data = dataHandler.DataHandler('data/imu_output.txt', 'data/ground_truth.txt', batch_size)
data = dataHandler.DataHandler(arguments.IMU_Data, arguments.Ground_Truth, batch_size)


# ####### #
# Network #
# ####### #

# input
network_input = tf.placeholder(tf.float64, shape=[None, data.input_size()])

# output from the network described in networkStructure.py
output = networkStructure.network(network_input)


# estimation
estimate = tf.placeholder(tf.float64, shape=[None, 7])

# --- network --- #




# cost function and optimization
cost = tf.reduce_sum(tf.pow(output - estimate, 2)) / (2 * data.training_data_size())
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# cost for the validation set
validation_cost = tf.reduce_sum(tf.pow(output - estimate, 2)) / (2 * data.validation_data_size())

# initialize session
init = tf.global_variables_initializer()

# training loop
with tf.Session() as sess:
    sess.run(init)

    print("Starting training with %d training samples and %d epochs" % (data.training_data_size(), epochs))

    for epoch in range(epochs):

        # train with all the training data
        while(data.training_data_available()):

            input_data, ground_truth = data.next_batch()
            sess.run(optimizer, {network_input: input_data, estimate: ground_truth})

        # reset training data iterator
        data.reset()

        # display intermediate results on the validation test set
        if(epoch % display_step == 0):
            full_data, full_ground_truth = data.full_training_data()
            training_cost = sess.run(cost, feed_dict={network_input: full_data, estimate: full_ground_truth})
            print("Epoch %d: %f" % (epoch, training_cost))


    print("Finished training")

    # test it on the validation set
    full_data, full_ground_truth = data.full_validation_data()
    training_cost = sess.run(validation_cost, feed_dict={network_input: full_data, estimate: full_ground_truth})

    print("Final Cost: %f" % training_cost)
