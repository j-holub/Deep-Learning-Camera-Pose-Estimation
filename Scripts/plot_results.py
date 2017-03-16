import matplotlib.pyplot as plt



# Argument parser
import argparse
# filesystem stuf
import os
import sys



# ################ #
# Argument Parsing #
# ################ #

parser = argparse.ArgumentParser()
parser.add_argument("results", help="Outputfile produced by the network")
arguments = parser.parse_args()

# check if the results file exists
if(not os.path.exists(arguments.results)):
    print("'%s' not found" % arguments.results)
    sys.exit()



# ############ #
# File Parsing #
# ############ #

epochs = []
training_cost = []
validation_cost = []

# parse the input file
result_file = open(os.path.abspath(arguments.results), "r")

for line in result_file.readlines()[1:]:

    ep, tc, vc = line.split(',')

    epochs.append(int(ep))
    training_cost.append(float(tc))
    validation_cost.append(float(vc))

result_file.close()



# ######## #
# Plotting #
# ######## #

# training data results
plt.plot(epochs, training_cost, label="Training Data")
# validation data results
plt.plot(epochs, validation_cost, label="Validation Data")

# labels
plt.xlabel('Epoch')
plt.ylabel('Cost')

# legend
plt.legend()


# save it
plt.savefig("result_plot.png")

# show it
plt.show()
