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
parser.add_argument("-o", "--output", help="Outputs the plot as a png file")
parser.add_argument("-nd", "--no-display", help="Does not display the plot", action='store_true')
parser.add_argument("-to", "--training-only", help="Only plots the training cost", action='store_true')
parser.add_argument("-vo", "--validation-only", help="Only plots the validation cost", action='store_true')
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
if(not arguments.validation_only):
    plt.plot(epochs, training_cost, label="Training Data")
# validation data results
if(not arguments.training_only):
    plt.plot(epochs, validation_cost, label="Validation Data")

# labels
plt.xlabel('Epoch')
plt.ylabel('Cost')

# legend
plt.legend()


# save it
if(arguments.output):
    plt.savefig(arguments.output, format="png")

# show it
if(not arguments.no_display):
    plt.show()
