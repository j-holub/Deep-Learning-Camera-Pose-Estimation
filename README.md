# Deep Learning Camera Pose Estimation

Camera Pose Estimation originally based on Kalman Filter, using Deep Learning and recurrent neural networks.

Based on the work of [Jason Rambach](https://av.dfki.de/members/rambach/).


# Usage

The program requires two input files, one holding the **imu measurements** and another one holding the **ground truth** data.


```
python <network_version.py> <imu_output> <ground_truth>
```

## Options

| Option         | Short Version | Type    | Meaning                                                               |
|----------------|---------------|---------|-----------------------------------------------------------------------|
| --batch-size   | -bs           | Integer | The Batch Size                                                        |
| --display-step | -ds           | Integer | How often the intermediate results should be output. Each **ds-th** epoch |
| --epochs       | -ep           | Integer | How many epochs to train                                              |
| --output       | -o            | String  | File to write the output to                                           |


# Plot Results

There is a script to plot the results called **plot_results.py** inside the *Scripts* folder.

It can plot the output files by simply calling

```
python plot_results.py <outputfile>
```

## Options

| Option            | Short Version | Type   | Meaning                                                                   |
|-------------------|---------------|--------|---------------------------------------------------------------------------|
| --no-display      | -nd           | Flag   | Does not display the plot. Useful when only the output file is desired    |
| --output          | -o            | String | Output file where the plot should be stored.Will be stored in png format |
| --training-only   | -to           | Flag   | Plots only the training cost                                              |
| --validation-only | -vo           | Flag   | Plots only the validation cost                                            |
| --model-output    | -mo           | Flag   | Output file where the model should be stored
|


## Technologies

This software is based on the [TensorFlow](https://www.tensorflow.org) Deep Learning Framework.

Unfortunately TensorFlows API has changed a lot here and there, sometimes documented often enough undocumented. That's why it may run with some version and won't with some others.

The version you will install depends on what OS and what distribution you use. It won't work with the latest **1.1** version but should work with something like **0.9**.

If you have veresion **0.10.0** you can easily run the code when you change the line 

```
init = tf.global_variables_initializer()
``` 

to

```
init= tf.initialize_all_varibles()
```

The latter is deprecated and no more available in later version, but the new version might not be available yet in an older version (like **0.10.0**).

## Literature

The original paper can be found [here](https://www.researchgate.net/publication/307410019_Learning_to_Fuse_A_Deep_Learning_Approach_to_Visual-Inertial_Camera_Pose_Estimation)
