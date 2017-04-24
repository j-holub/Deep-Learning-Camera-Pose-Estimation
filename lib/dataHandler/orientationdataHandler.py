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
        self.avrg_number_of_measurements = 1

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
                read_values.append(np.asarray(values))

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
        measurement_sizes = [(len(measurement)/6) for measurement in read_values]
        self.avrg_number_of_measurements = int(np.around(np.mean(measurement_sizes)) * 6)


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
            # add the ground_truth to the labels
            read_ground_truth.append(np.asarray(ground_truth))



        # ################ #
        # Process the Data #
        # ################ #

        # split the data into positions and quanterion information
        positions      = [gt[0:3] for gt in read_ground_truth]
        quanterionInfo = [gt[3:]  for gt in read_ground_truth]


        # create the displacement information
        read_ground_truth = []
        for i in range(0, len(positions)-1):
            gt = []
            # displacement
            gt = positions[i+1] - positions[i]
            # quanterionInfo
            gt = np.append(gt, quanterionInfo[i+1])
            # gt = [x, y, z, q_0, q_1, q_2, q_3]
            read_ground_truth.append(gt)


        # drop the last one
        read_values = read_values[:-1]

        # append the quantierion information to the input
        for i in range(0, len(read_values)):
            read_values[i] = np.append(read_values[i], quanterionInfo[i])



        read_values_filtered = []
        read_ground_truth_filtered = []

        # remove entries that are longer or shorter than the average
        for i in range(len(read_values)):
            # only use measurements with enough values
            if(len(read_values[i])-4 >= self.avrg_number_of_measurements):

                # trim the array if it is too long, leaving the last 4 (quanterion information) in tact
                if(len(read_values[i])-4 > self.avrg_number_of_measurements):
                    delete_indeces = np.arange(len(read_values[i]) - self.avrg_number_of_measurements - 4) + self.avrg_number_of_measurements
                    read_values[i] = np.delete(read_values[i], delete_indeces)

                # append them to  the filtered version
                read_values_filtered.append(read_values[i])
                read_ground_truth_filtered.append(read_ground_truth[i])


        # ####################### #
        # test & validation split #
        # ####################### #

        # Split the data into train and validation set
        trainingAmount = int(math.ceil((len(read_values_filtered) / 100.0) * 70))

        # training set
        self.training_data = read_values_filtered[0:trainingAmount]
        self.training_ground_truth = read_ground_truth_filtered[0:trainingAmount]

        # validation set
        self.validation_data = read_values_filtered[trainingAmount:]
        self.validation_ground_truth = read_ground_truth_filtered[trainingAmount:]


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
        data   = np.reshape(data, [temp_batch_size, self.avrg_number_of_measurements+4])

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
        return self.avrg_number_of_measurements+4

    # ------------- #
    # Training Data #
    # ------------- #

    # returns the full training data and training ground truth
    # return type: (nparray(size, 67), nparray(size, 7))
    def full_training_data(self):

        data = np.reshape(self.training_data, [len(self.training_data), self.avrg_number_of_measurements+4])
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

        data = np.reshape(self.validation_data, [len(self.validation_data), self.avrg_number_of_measurements+4])
        labels = np.reshape(self.validation_ground_truth, [len(self.validation_ground_truth), 7])

        return (data, labels)


    # returns the total number of validation samples
    # return type: int
    def validation_data_size(self):
        return len(self.validation_data)
