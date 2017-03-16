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
