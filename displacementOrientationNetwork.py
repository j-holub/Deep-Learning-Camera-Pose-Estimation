# TensorFlow
import tensorflow as tf

# Argument parser
import argparse
# filesystem stuf
import os
import sys

# DataHandler
from lib.dataHandler.displacementOrientationDataHandler import DataHandler
# Network
from lib.networks.generalNetwork import Network


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
parser.add_argument("--output", "-o")
parser.add_argument("--model_output", "-mo")

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
try:
    epochs = int(arguments.epochs)
    display_step = int(arguments.display_step)
    batch_size = int(arguments.batch_size)
except ValueError as e:
    print("Epochs, Display Step and Batch Size have to be postitve integral numbers")
    sys.exit()


if(arguments.output):
    # base path
    path = os.path.dirname(arguments.output)
    # create the path if it does not exist
    if(not os.path.exists(path)):
        os.mkdir(path)

    output_file = open(arguments.output, "w")


if(arguments.model_output):
    # base path
    path = os.path.dirname(arguments.model_output)
    # create the path if it does not exist
    if(not os.path.exists(path)):
        os.mkdir(path)


# #### #
# Data #
# #### #

# set up the data
data = DataHandler(arguments.IMU_Data, arguments.Ground_Truth, batch_size)


# ####### #
# Network #
# ####### #

# input
network_input = tf.placeholder(tf.float64, shape=[None, data.input_size()])

# output from the network described in networkStructure.py
output = Network(network_input)


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


# load the saver if model is to be saved
if(arguments.model_output):
    saver = tf.train.Saver()



# training loop
with tf.Session() as sess:
    sess.run(init)

    print("Starting training with %d training samples, %d epochs and a batch size of %d" % (data.training_data_size(), epochs, batch_size))
    print("Epoch: Training Cost, Validation Cost")

    if(arguments.output):
        output_file.write("Epoch, Training Cost, Validation Cost\n")

    for epoch in range(epochs+1):

        # train with all the training data
        while(data.training_data_available()):

            input_data, ground_truth = data.next_batch()
            sess.run(optimizer, {network_input: input_data, estimate: ground_truth})

        # reset training data iterator
        data.reset()

        # display intermediate results on the training and the validation set
        if(epoch % display_step == 0):
            # complete training data
            full_data, full_ground_truth = data.full_training_data()
            # comlete validation data
            full_validation_data, full_validation_ground_truth = data.full_validation_data()
            _, training_cost   = sess.run([optimizer, cost], feed_dict={network_input: full_data, estimate: full_ground_truth})
            _, validation_test_cost = sess.run([optimizer, validation_cost], feed_dict={network_input: full_validation_data, estimate: full_validation_ground_truth})
            print("%d: %f, %f" % (epoch, training_cost, validation_test_cost))
            if(arguments.output):
                output_file.write("%d, %f, %f\n" % (epoch, training_cost, validation_test_cost))


    if(arguments.model_output):
        save_path = saver.save(sess, os.path.abspath(arguments.model_output))
        print("Saved model to '%s'" % save_path)


    print("Finished training")
