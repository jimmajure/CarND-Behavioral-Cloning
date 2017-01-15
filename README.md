# CarND-Behavioral-Cloning

## Training Data

Training data were collected using the simulator in training mode. The key was to collect appropriate data to train the model to recover when the car moves off of the center of the roadway. The suggestion in the instructions was to turn the camera on and off when the car was moving from the edges to the center of the road. In practice this was quite difficult due to the ergonomics of the simulator.

Therefore, instead of trying to capture recovery images by driving from the edge to the road to the center, I captured entire laps while driving on the left-hand side of the road and on the right-hand side of the road and saved them into separate folders. This is shown in the figure below. While training, I then adjusted the steering angles for all images captured on the left or right side of the road to 
