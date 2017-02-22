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


        # the average number of measurements per data
        self.input_size = 1

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

            # get rid of line breaks
            line = line.strip()

            # if image was found in the line it marks the end of the meassurement squence
            # for that image
            if('image' in line):
                # old data case
                if(len(line.split(' ')) == 1):
                    label = int(line[5:].strip())
                # new data case
                else:
                    label = int(line.split(' ')[1])

                # store all the read data
                data[label] = values

                # reset buffer
                values = []

            # this is almost always a line of <timestamp> m1 m2 m3 m4 m5 m6
            # sometimes it's a newline character
            elif(line != '\n'):
                line_split = line.split(' ')

                # new data case
                if(line_split[0] == 'imu'):
                    # add all numerical values (except the timestamp) to the value list
                    values.extend([float(value) for value in line_split[2:] if value != ''])
                # old data case
                else:
                    # add all numerical values (except the timestamp) to the value list
                    values.extend([float(value) for value in line_split[1:] if value != ''])

            # read the next line
            line = imu_file.readline()



        # get the average number of measurements to detect which
        # values to keep
        measurement_sizes = [len(data[label])/6 for label in data.keys()]
        self.input_size = int(np.around(np.mean(measurement_sizes)) * 6)


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
                read_ground_truth.append(ground_truth)


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


        values = []
        ground_truth = []

        print self.input_size

        # remove entries that are longer or shoter than the average
        for i in range(len(read_values)):
            # only use measurements with enough values
            if(len(read_values[i])-7 >= self.input_size):

                # trim the array if it is too long, leaving the last 7 (past pose) in tact
                if(len(read_values[i])-7 > self.input_size):
                    delete_indeces = np.arange(len(read_values[i]) - self.input_size - 7) + self.input_size
                    read_values[i] = np.delete(read_values[i], delete_indeces)

                # append them
                values.append(read_values[i])
                ground_truth.append(read_ground_truth[i])




        # ####################### #
        # test & validation split #
        # ####################### #

        # Split the data into train and validation set
        trainingAmount = int(math.ceil((len(values) / 100.0) * 70))

        # training set
        self.training_data = values[0:trainingAmount]
        self.training_ground_truth = ground_truth[0:trainingAmount]

        # validation set
        self.validation_data = values[trainingAmount:]
        self.validation_ground_truth = ground_truth[trainingAmount:]


    # --------------------------------------------------------------------------


    # Loads the next batch of training data according to the batch size
    # return type: tuple of data and label matrix
    def next_batch(self):

        # check if the batch is bigger than the remaining data
        if(self.batch_pointer + self.batch_size <= len(self.training_ground_truth)):
            upperbound = self.batch_pointer + self.batch_size
            temp_batch_size = self.batch_size
        else:
            upperbound = len(self.training_data)
            temp_batch_size = upperbound - self.batch_pointer

        # data matrix
        data   = self.training_data[self.batch_pointer : upperbound]
        data   = np.reshape(data, [temp_batch_size, 67])

        # label matrix
        labels = self.training_ground_truth[self.batch_pointer : upperbound]
        labels = np.reshape(labels, [temp_batch_size, 7])

        self.batch_pointer += self.batch_size

        return (data, labels)


    # resets the batch pointer to start anew
    # return type: void
    def reset(self):
        self.batch_pointer = 0

    # returns the size of the data (#meassurements*6 + 7)
    # return type: int
    def input_size(self):
        return self.input_size+7

    # ------------- #
    # Training Data #
    # ------------- #

    # returns the full training data and training ground truth
    # return type: (nparray(size, 67), nparray(size, 7))
    def full_training_data(self):

        data = np.reshape(self.training_data, [len(self.training_data), 67])
        labels = np.reshape(self.training_ground_truth, [len(self.training_ground_truth), 7])

        return (data, labels)


    # check if there is still some data to load an process
    # return type: bool
    def training_data_available(self):
        return self.batch_pointer < len(self.training_data)


    # returns the total number of training samples
    # return type: int
    def training_data_size(self):
        return len(self.training_data)

    # --------------- #
    # Validation Data #
    # --------------- #

    # returns the full validation data and training ground truth
    # return type: (nparray(size, 67), nparray(size, 7))
    def full_validation_data(self):

        data = np.reshape(self.validation_data, [len(self.validation_data), 67])
        labels = np.reshape(self.validation_ground_truth, [len(self.validation_ground_truth), 7])

        return (data, labels)


    # returns the total number of validation samples
    # return type: int
    def validation_data_size(self):
        return len(self.validation_data)
