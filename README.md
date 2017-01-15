# CarND-Behavioral-Cloning

## Training Data

Training data were collected using the simulator in training mode. There are two aspects of the data to consider for successful fitting of a model to predict steering angle:

1. The inclusion of appropriate recovery training data
1. The distribution the steering angle data in the training data

### Recovery Data
In order to build a model that successfully guides the car around the track, it is important to have data that tells the model how to recover when the car approaches the roadway sides. The suggestion in the instructions was to turn the image capture on and off when the car was moving from the edges to the center of the road. In practice this was quite difficult due to the ergonomics of the simulator.

Instead of trying to capture recovery images by driving from the edge to the road to the center, I captured entire laps while driving on the left-hand side of the road and on the right-hand side of the road and saved them into separate folders. This is shown in the figure below.


![alt tag](readme_images/roadway.png)
**Figure 1: positions on the roadway used to collect training data.**

With the training data from the left-hand (or right-hand) side of the road in a separate data set, I then adjusted the steering angle for each of the images to simulate recovering from the side of the roadway back to the middle of the roadway.

### Steering Angle Distribution

A key insight of the training data that helped me progress in building a successful model was that most of the training data contained steering angles of near 0.0 values. The large number of training points with steering angle near 0.0 dominated the learning and made it difficult for the model to "learn" to manuver around sharp corners. (This insight was taken from the forums in a post by Milutin Nikolic). The distibution of steering angle values from a sample data set is shown in the figure below.

**figure 2**

To compensate for this, I filtered out a given percentage of the values near zero when generating data for the model fit runs.

### Data Generator

I used a data generator to facilitate both the recovery data and the filtering of near-zero values. The generator constructor method is shown here:

```
SimulatorGenerator(data, include_params)
```

The data parameter allows one or more datasets to be specified as inputs to the generator. The generator expects the data parameter to contain a list of tuples. Each tuple contains a the name of a directory and a list of tuples each of which contains a camera and a steering adjustment. Here is an example of specifying three data directories, one with data from the center of the road, one with data from the left-hand side of the road, and one with data from the right-hand side of the road. Cameras that are to the either side of the road center are adjusted to direct the car back to the road center.

```
data = [
    ("/home/jim/workspace/drive_data_center_2",[('left',0.05),('center',0.0),('right',-0.05)]),
    ("/home/jim/workspace/drive_data_left_2",[('left',0.32),('center',0.3),('right',0.25)]),
    ("/home/jim/workspace/drive_data_right_2",[('left',-0.25),('center',-0.3),('right',-0.32)]),
    ]
```

The include_params parameter is a tuple that contains the value defining "near-zero" and a percentage of near-zero values to be included in the training/validation data. Here is an example that specifies that 15% of the data with steering angles with an absolute value less than or equal to 0.01 should be included in the training data. The near-zero values selected were taken from a uniform random distribution.

```
include_params=(0.01,15)
```
The generator class builds a generator for both training and validation according to the following steps:

1. process the data in each data set and retain the specified percentage of near-zero values
1. for each remaining data point, add a value for each of the cameras specified, adjusting the steering angle by the given values
1. shuffle/split the data, retaining 30% of the values for validation

An example of using the generator is shown here.
```
hist = mdl.fit_generator(generator=generator.train_generator(), 
                samples_per_epoch=generator.get_train_size(),
                nb_epoch=50,
                verbose=2,
                validation_data=generator.validation_generator(),
                nb_val_samples=generator.get_validation_size(),
                callbacks=[EarlyStopping('val_loss', 0.001, 1)])
```
## Models

Several models were considered beginning with an implementation of the model in the NVIDIA paper, End to End Learning for Self-Driving Cars. The core model included the following layers:

1. 3 5x5 convolution layers
1. 2 3x3 convolution layers
1. 3 fully connected dense layers

The models considered are described in the following table.

|Model|Description|
|---|---|
|```model1```|This is the basic network from the NVIDIA paper. The images are expected to have dimensions of 80x160, i.e., preprocessing is expected to resize the images from the original 160x320.|
|```model2```|This is ```model1``` with DropOut layers inserted after the 1st 5x5 convolution layer, after the 3rd 5x5 convolution layer, and after the 1st fully connected layer.|
|```model4```|This is ```model2``` but expecting image dimensions of 160x320 and with a MaxPooling layer added at the beginning of the model that reduces the image dimensions to 80x160|
