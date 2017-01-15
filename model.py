'''
Created on Jan 7, 2017

@author: jim
'''
from csv import DictWriter, DictReader
from datetime import datetime
import gc
import os.path
import random
import time

import cv2
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from sklearn.utils import shuffle

import numpy as np
from sklearn.model_selection import ShuffleSplit


Yscale = [0.299, 0.587, 0.114]
Uscale = [-0.14713, -0.28886, 0.436]
Vscale = [0.615,-0.51499, -0.10001]

YUVscale = [Yscale, Uscale, Vscale]


def rgb2yuv(rgb):
    '''
    Transform an RGB image to a YUV image.
    '''
    return np.dot(rgb, YUVscale)
    

def scale(x, range=[0.0,1.0]):
    '''
    Scale image data band-wise to the specified range.
    '''
    assert len(x.shape) == 3, "Expecting 3 dimensional array, got {} dimensional array.".format(len(x.shape))
    channel_1 = x[:,:,0:1]
    channel_2 = x[:,:,1:2]
    channel_3 = x[:,:,2:3]

    pct = lambda x: 1.*(x-np.min(x))/(np.max(x)-np.min(x))
    scale = lambda percent: range[0]+percent*(np.max(range)-np.min(range))
    
    return np.concatenate((scale(pct(channel_1)), scale(pct(channel_2)), scale(pct(channel_3))), axis=2)
        

def load_image(imagename):
    '''
    Read an image from the disk and apply the image adjustment.
    '''
    img = cv2.imread(imagename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return adjust_image(img)

def adjust_image(img):
    '''
    Adjust an RGB image for inclusion in the model.
    '''
    img = scale(rgb2yuv(img), range=[-0.5,0.5])
    return img

def batch(X, y, batch_size):
    '''
    Yield batches of the provided arrays of size batch_size.
    '''
    for offset in range(0, len(X), batch_size):
        end = offset + batch_size
        batch_x, batch_y = X[offset:end], y[offset:end]
        yield batch_x, batch_y
        
def relativize_image_name(name, location):
    '''
    The data collected by the tool and the data provided by Udacity
    have different pathnames for the image files. This method takes
    a path, strips the file name and pre-pends it with a specified 
    directory name.
    '''
    return location+"/"+name.split("/")[-1]

def should_include(angle, params):
    '''
    Boolean function to indicate whether to include a data point should be included.
    
    angle - an angle to be considered
    params - a tuple containing the cutoff above which to include an data point
        and a percent (0-99) of points to include that are below the cutoff
    '''
    if not params:
        return True
    else:
        return np.abs(angle)>params[0] or random.randint(0,99) < params[1]

class SimulatorGenerator():
    '''
    A class to create training and validation generators suitable for use
    in Keras model training.
    '''
    def __init__(self, data, cameras=(0.005,['center']), include_params=None):
        # a list to contain all of the final data points added to the training/validation sets
        self.__data = []
        # a list to store the original steering angle data for distribution plotting
        self.original_data = []
        for data_spec in data:
            dir = data_spec[0]
            assert os.path.isdir(dir), "Directory, {}, does not exist.".format(dir)
            image_dir = dir + "/IMG"
            assert os.path.isdir(image_dir),  "Image subdirectory, {}, does not exist.".format(image_dir)
            driving_log = dir + "/driving_log.csv"
            assert os.path.isfile(driving_log),  "Driving log file, {}, does not exist.".format(driving_log)
            
            cameras = data_spec[1]
            for cam in cameras:
                assert cam[0] in ['left','center','right']
            
    
            with open(driving_log,'r') as f:
                reader = DictReader(f, fieldnames=['center','left','right','angle','throttle','brake','speed'])
                for l in reader:
                    self.original_data.append(float(l['angle']))
                    if should_include(float(l['angle']), include_params):
                        # if we include this point, then process each specified camera...
                        for camera in cameras:
                            self.__data.append((relativize_image_name(l[camera[0]], image_dir),
                                                float(l['angle'])+camera[1]))
        
        self.__data = np.array(self.__data)
        self.__train_size = -1
        self.__validation_size = -1

        self.__train_data = []
        self.__validate_data = []
#         self.validate = []
        shuffler = ShuffleSplit(1, test_size=0.3)
        for train_index, val_index in shuffler.split(self.__data):
            self.__train_data = self.__data[train_index]
            self.__validate_data = self.__data[val_index]
            
        
        self.reshuffle()

        print("Driving log contains {} lines.".format(len(self.__data)))
        
    def get_train_size(self):
        return self.__train_size
    
    def get_validation_size(self):
        return self.__validation_size
    
    def train_generator(self):
        while True:
            # Each time through the data, let's re-shuffle
            self.reshuffle()
            for d in self.train:
                images = d[0]
                angles = d[1]
                yield (np.array([load_image(image_name) for image_name in images]), angles)
                        
    def validation_generator(self):
        while True:
            for d in self.validate:
                images = d[0]
                angles = d[1]
                yield (np.array([load_image(image_name) for image_name in images]), angles)

    def reshuffle(self):
        self.__train_data = shuffle(self.__train_data)
        self.__validate_data = shuffle(self.__validate_data)
            
        self.train = []
        self.validate = []
        
        # now split the data into batches...
        self.__train_size = 0
        for bx, by in batch(np.array([i for (i,_) in self.__train_data]), np.array([a for (_,a) in self.__train_data]), batch_size=128):
            self.train.append((bx, by))
            self.__train_size += len(bx)
            
        self.__validation_size = 0
        for bx, by in batch(np.array([i for (i,_) in self.__validate_data]), np.array([a for (_,a) in self.__validate_data]), batch_size=128):
            self.validate.append((bx, by))
            self.__validation_size += len(bx)

    def plot_angle_distribution(self):
        import matplotlib.pyplot as plt
        plt.hist(np.array(self.original_data), 200, normed=1, facecolor='green', alpha=0.75)
        plt.show()


def model1():
    '''
    The basic model described in the NVIDIA paper, End to End Learning for Self-Driving Cars.
    Input Dimensions: (80,160,3)
    '''
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=(80, 160, 3), subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', input_shape=(38, 78, 24), subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', input_shape=(17, 37, 36), subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(6, 16, 48), subsample=(1,1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(4, 14, 48), subsample=(1,1), activation='relu'))
    # input = 48*4*14=2688
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model

def model2():
    '''
    This model extends model to include dropout layers between sections of the model.
    Input Dimensions: (80,160,3)
    '''
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=(80, 160, 3), subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(Flatten())    
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model

def model4():
    '''
    This model extends model2 to include a max pooling step at the beginning of the model. This 
    step reduces the size of the image by a factor of 2.
    Input Dimensions: (160,320,3)
    '''
    model = Sequential()
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode="valid", input_shape=(160, 320, 3), ))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model

def model5():
    '''
    Extends model4 by including MaxPooling steps between sections of the model.
    Input Dimensions: (160,320,3)
    '''
    model = Sequential()
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode="valid", input_shape=(160, 320, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode="valid"))
    model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode="valid"))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode="valid"))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model

def model6():
    '''
    Extends model5 by including BatchNormilization steps after each convolution layer.
    Input Dimensions: (160,320,3)
    '''
    model = Sequential()
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode="valid", input_shape=(160, 320, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode="valid"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode="valid"))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode="valid"))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model

def model7():
    '''
    Extends model6 by including l2 weight normalization after each convolution or dense layer.
    Input Dimensions: (160,320,3)
    '''

    model = Sequential()
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode="valid", input_shape=(160, 320, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu', W_regularizer="l2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode="valid"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu', W_regularizer="l2"))
    model.add(BatchNormalization())
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu', W_regularizer="l2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode="valid"))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu', W_regularizer="l2"))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu', W_regularizer="l2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode="valid"))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100, activation='relu', W_regularizer="l2"))
    model.add(Dense(50, activation='relu', W_regularizer="l2"))
    model.add(Dense(10, activation='relu', W_regularizer="l2"))
    model.add(Dense(1))

    return model


def train_model(model, data, cameras, include_params=None, write_results=False):
    '''
    Execute a training run...
    param cameras is depricated.
    '''
    mdl = model()
    generator = SimulatorGenerator(data,cameras,include_params)
    mdl.compile(optimizer='adam',loss="mean_squared_error", metrics=['mean_absolute_error'])
    
    start_time = time.time()

    hist = mdl.fit_generator(generator=generator.train_generator(), 
                        samples_per_epoch=generator.get_train_size(),
                        nb_epoch=50,
                        verbose=2,
                        validation_data=generator.validation_generator(),
                        nb_val_samples=generator.get_validation_size(),
                        callbacks=[EarlyStopping('val_loss', 0.001, 1)])
    
    with open("model_out.json", 'w') as out:
        out.write(mdl.to_json())
    
    gc.collect()
    mdl.save_weights("model_out.h5")
    
    # save the results to a csv file for posterity...
    if write_results:
        with open('results.csv','a') as results:
            writer = DictWriter(results,fieldnames = ['Date','Data Source',
                                                       'Model','Camera','Include Params','Epochs','Min Loss',
                                                       'Training Samples','Validation Samples','Time (s)'])
            writer.writerow({'Date': str(datetime.now()),
                             'Data Source': [(d.split('/')[-1],a) for (d,a) in data],
                             'Model':model.__name__,
                             'Camera':cameras,
                             'Include Params': include_params,
                             'Epochs':hist.epoch[-1]+1,
                             'Min Loss':np.min(hist.history['val_loss']),
                             'Training Samples': generator.get_train_size(),
                             'Validation Samples':generator.get_validation_size(),
                             'Time (s)':"{:.3}".format(time.time()-start_time)})

if __name__ == '__main__':
    data = [
        ("/home/jim/workspace/drive_data_center_2",[('left',0.05),('center',0.0),('right',-0.05)]),
        ("/home/jim/workspace/drive_data_left_2",[('left',0.32),('center',0.3),('right',0.25)]),
        ("/home/jim/workspace/drive_data_right_2",[('left',-0.25),('center',-0.3),('right',-0.32)]),
        ]
#     data = [
#         ("/home/jim/workspace/drive_data_center_2",[('left',0.1),('center',0.0),('right',-0.1)]),
#         ("/home/jim/workspace/drive_data_left_2",[('left',0.35),('center',0.3),('right',0.2)]),
#         ("/home/jim/workspace/drive_data_right_2",[('left',-0.35),('center',-0.3),('right',-0.2)]),
#         ]
#     data = [
#         ("/home/jim/workspace/drive_data_center_2",[('center',0.0)]),
#         ("/home/jim/workspace/drive_data_left_2",[('center',0.3)]),
#         ("/home/jim/workspace/drive_data_right_2",[('center',-0.3)]),
#         ]
    train_model(model6, data, cameras=None, 
                include_params=(0.01,15), write_results=False)
