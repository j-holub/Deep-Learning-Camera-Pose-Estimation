import math
import numpy as np
import tensorflow as tf

class DataHandler:



    def __init__(self, imu_output, ground_truth, batch_size):

        # member vars
        # training set
        self.training_data = []
        self.training_ground_truth = []

        # validation set
        self.validation_data = []
        self.validation_ground_truth = []


        self.batch_size = batch_size
        self.batch_pointer = 0

        # -------

        # container to hold the read data
        read_values       = []
        read_ground_truth = []

        # dict to store the meassurements
        data = {}

        # ###################### #
        # prepare the imu output #
        # ###################### #

        # open the file
        imu_file = open(imu_output, 'r')
        line = imu_file.readline()

        # temporary store the values of one messurement
        values = []

        # label blacklist
        # some datasets do not have enough measurements, these have to be ignored
        blacklist = []

        # parse the file
        while(line != ""):


            # if image was found in the line it marks the end of the meassurement squence
            # for that image
            if('image' in line):
                label = int(line[5:].strip())
                # sometimes there are too many meassurements
                # this ensures that only 10 meassurements (6 values each)
                # are taken
                # sometimes there are not enough meassurements, these have to be ignored
                if(len(values[:60]) == 60):
                    data[label] = values[:60]
                else:
                    blacklist.append(label)
                values = []
            # this is almost always a line of <timestamp> m1 m2 m3 m4 m5 m6
            # sometimes it's a newline character
            elif(line != '\n'):
                line_split = line.split(' ')
                # add all numerical values (except the timestamp) to the value list
                values.extend([float(value) for value in line_split[1:] if value != ''])

            # read the next line
            line = imu_file.readline()


        # ############################# #
        # prepare the ground_truth file #
        # ############################# #


        # read the file
        gt_lines = open(ground_truth, 'r').readlines()
        # parse it
        for line in gt_lines:
            # Format: label d1 d2 d3 d4 d5 d6 d7
            line_split = line.split(' ')
            # ground truth values (without label)
            ground_truth = [float(value) for value in line_split[1:]]
            # this adds the gt to the meassurements for the network input as stated in the paper
            label = int(line_split[0])
            if(label not in blacklist):
                data[label].extend(ground_truth)
                # add the ground_truth to the labels
                read_ground_truth.append(np.asarray(ground_truth))


        # parse the information from the files into the label and value array
        for label in sorted(data.keys()):
            data[label] = np.asarray(data[label])
            read_values.append(data[label])

        # values [n] has to correspond to labels [n+1]
        # pose prediction

        # drop the first
        read_ground_truth = read_ground_truth[1:]
        # drop the last
        read_values = read_values[:-1]


        # ####################### #
        # test & validation split #
        # ####################### #

        # Split the data into train and validation set
        trainingAmount = int(math.ceil((len(read_values) / 100.0) * 70))
        print trainingAmount

        # training set
        self.training_data = read_values[0:trainingAmount]
        self.training_ground_truth = read_ground_truth[0:trainingAmount]

        # validation set
        self.validation_data = read_values[trainingAmount:]
        self.validation_ground_truth = read_ground_truth[trainingAmount:]

        print(len(self.training_data))
        print(len(self.training_ground_truth))

    # --------------------------------------------------------------------------


    # Loads the next batch of training data according to the batch size
    # return type: tuple of data and label matrix
    def next_batch(self):

        # check if the batch is bigger than the remaining data
        if(self.batch_pointer + self.batch_size <= len(self.training_ground_truth)):
            upperbound = self.batch_pointer + self.batch_size
        else:
            upperbound = len(self.training_data)

        # data matrix
        data   = self.training_data[self.batch_pointer : self.batch_pointer + self.batch_size]
        data   = np.reshape(data, [self.batch_size, 67])

        # label matrix
        labels = self.training_ground_truth[self.batch_pointer : self.batch_pointer + self.batch_size]
        labels = np.reshape(labels, [self.batch_size, 7])

        self.batch_pointer += self.batch_size

        return (data, labels)

    # returns the full date and labels reshaped to (size, 67) and (size, 7)
    # return type: (nparray(size, 67), nparray(size, 7))
    def full_data(self):

        data = np.reshape(self.values, [len(self.values), 67])
        labels = np.reshape(self.labels, [len(self.labels), 7])

        return (data, labels)

    # resets the batch pointer to start anew
    # return type: void
    def reset(self):
        self.batch_pointer = 0

    # check if there is still some data to load an process
    # return type: bool
    def training_data_available(self):
        return self.batch_pointer < len(self.training_data)

    # returns the total number of training samples
    # return type: int
    def training_data_size(self):
        return len(self.training_data)
