# Deep Learning Camera Pose Estimation

Camera Pose Estimation originally based on Kalman Filter, using Deep Learning and recurrent neural networks.

Based on the work of [Jason Rambach](https://av.dfki.de/members/rambach/).


# Usage

The program requires two input files, one holding the **imu measurements** and another one holding the **ground truth** data.


```
python network.py <imu_output> <ground_truth>
```

## Options

| Option         | Short Version | Type    | Meaning                                                               |
|----------------|---------------|---------|-----------------------------------------------------------------------|
| --batch-size   | -bs           | Integer | The Batch Size                                                        |
| --display-step | -ds           | Integer | How often the intermediate results should be output. Each ds-th epoch |
| --epochs       | -ep           | Integer | How many epochs to train                                              |
| --output       | -o            | String  | File to write the output to                                           |


## Technologies

This software is based on the [TensorFlow](https://www.tensorflow.org) Deep Learning Framework.

## Literature

The original paper can be found [here](https://www.researchgate.net/publication/307410019_Learning_to_Fuse_A_Deep_Learning_Approach_to_Visual-Inertial_Camera_Pose_Estimation)
