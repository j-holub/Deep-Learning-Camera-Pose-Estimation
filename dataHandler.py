
class DataHandler:



    def __init__(self, imu_output, ground_truth, batch_size):

        # member vars
        self.values = []
        self.labels = []

        self.batch_size = batch_size
        self.batch_pointer = 0


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

            line = imu_file.readline()

            # if image was found in the line it marks the end of the meassurement squence
            # for that image
            if('image' in line):
                label = int(line[5:].strip())
                # sometimes there are too many meassurements
                # this ensures that only 10 meassurements (6 values each)
                # are taken
                data[label] = values[:60]
                values = []
            # this is almost always a line of <timestamp> m1 m2 m3 m4 m5 m6
            # sometimes it's a newline character
            elif(line != '\n'):
                line_split = line.split(' ')
                # add all numerical values (except the timestamp) to the value list
                values.extend([float(value) for value in line_split[1:] if value != ''])



        # ############################# #
        # prepare the ground_truth file #
        # ############################# #


        # read the file
        gt_lines = open(ground_truth, 'r').readlines()
        # parse it
        for line in gt_lines:
            # Format: label d1 d2 d3 d4 d5 d6 d7
            line_split = line.split(' ')
            # this adds the gt to the meassurements for the network input as stated in the paper
            data[int(line_split[0])].extend([float(value) for value in line_split[1:]])


        # parse the information from the files into the label and value array
        for label in sorted(data.keys()):
            self.labels.append(label)
            self.values.append(data[label])


    # --------------------------------------------------------------------------


    # Loads the next batch of training data according to the batch size
    # parameters
    #   data_identifier - how the data is called in the feed dict
    #   label_identifier - how the label is called in the feed dict
    # return type: dict [data, labels]
    def next_batch(self, data_identifier, label_identifier):
        batch = {}

        # check if the batch is bigger than the remaining data
        if(self.batch_pointer + self.batch_size <= len(self.labels)):
            upperbound = self.batch_pointer + self.batch_size
        else:
            upperbound = len(self.values)

        batch[data_identifier]   = self.values[self.batch_pointer : self.batch_pointer + self.batch_size]
        batch[label_identifier] = self.labels[self.batch_pointer : self.batch_pointer + self.batch_size]

        self.batch_pointer += self.batch_size

        return batch

    # resets the batch pointer to start anew
    # return type: void
    def reset(self):
        self.batch_pointer = 0

    # check if there is still some data to load an process
    # return type: bool
    def data_available(self):
        return self.batch_pointer < len(self.labels)

    # returns the total number of training samples
    # return type: int
    def training_data_size(self):
        return len(self.values)
