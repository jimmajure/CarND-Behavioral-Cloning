# CarND-Behavioral-Cloning

## Training Data

Training data were collected using the simulator in training mode. There are two aspects of the data to consider for successful fitting of a model to predict steering angle:

1. The inclusion of appropriate recovery training data
1. The distribution the steering angle data in the training data

### Recovery Data
In order to build a model that successfully guides the car around the track, it is important to have data that tells the model how to recover when the car approaches the roadway sides. The suggestion in the instructions was to turn the image capture on and off when the car was moving from the edges to the center of the road. In practice this was quite difficult due to the ergonomics of the simulator.

Instead of trying to capture recovery images by driving from the edge to the road to the center, I captured entire laps while driving on the left-hand side of the road and on the right-hand side of the road and saved them into separate folders. This is shown in the figure below.

**figure 1**

With the training data from the left-hand (or right-hand) side of the road in a separate data set, I then adjusted the steering angle for each of the images to simulate recovering from the side of the roadway back to the middle of the roadway.

### Steering Angle Distribution

A key insight of the training data that helped me progress in building a successful model was that most of the training data contained steering angles of near 0.0 values. The large number of training points with steering angle near 0.0 dominated the learning and made it difficult for the model to "learn" to manuver around sharp corners. (This insight was taken from the forums in a post by Milutin Nikolic in the forums). The distibution of steering angle values from a sample data set is shown in the figure below.

**figure 2**

To compensate for this, I filtered out a given percentage of the values near zero when generating data for the model fit runs.

### Data Generator

I used a data generator to facilitate both the recovery data and the filtering of near-zero values. The 
