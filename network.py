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
epochs = 100
display_step = 10
learning_rate = 0.01

# --- Basic Setup --- #


# ####### #
# Network #
# ####### #

# input
network_input = tf.placeholder(tf.float64, shape=[None, input_size])

# output from the network described in networkStructure.py
output = networkStructure.network(network_input)


# estimation
estimate = tf.placeholder(tf.float64, shape=[None, 7])

# --- network --- #


# set up the data
training_data = dataHandler.DataHandler('data/imu_output.txt', 'data/ground_truth.txt', batch_size)



# cost function and optimization
cost = tf.reduce_sum(tf.pow(output - estimate, 2)) / (2 * training_data.training_data_size())
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# initialize session
init = tf.global_variables_initializer()

# training loop
with tf.Session() as sess:
    sess.run(init)

    print("Starting training with %d training samples and %d epochs" % (training_data.training_data_size(), epochs))

    for epoch in range(epochs):

        # train with all the training data
        while(training_data.training_data_available()):

            data, labels = training_data.next_batch()
            sess.run(optimizer, {network_input: data, estimate: labels})

        # reset training data iterator
        training_data.reset()

        # display intermediate results
        if(epoch % display_step == 0):
            full_data, full_labels = training_data.full_data()
            error = sess.run(cost, feed_dict={network_input: full_data, estimate: full_labels})
            print("Epoch %d: %f" % (epoch, error))


    print("Finished training")

    full_data, full_labels = training_data.full_data()
    error = sess.run(cost, feed_dict={network_input: full_data, estimate: full_labels})

    print("Final Cost: %f" % error)
