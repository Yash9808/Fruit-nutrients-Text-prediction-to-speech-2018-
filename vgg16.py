#!/usr/bin/env python
from __future__ import division, print_function
import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
import h5py

# path = 'data/sample/'
path = 'data/'

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.
        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr

model = Sequential()
model.add(Lambda(vgg_preprocess, input_shape = (100, 100, 3), output_shape = (100, 100, 3)))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation = 'softmax'))

model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
fname='model.h5'
model.load_weights(fname)

model.pop()
for layer in model.layers: 
    layer.trainable = False

model.add(Dense(83, activation = 'softmax'))
model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

datagen = image.ImageDataGenerator()
trn_batches = datagen.flow_from_directory(path + 'training/', target_size = (100, 100),
            class_mode = 'categorical', shuffle = True, batch_size = 10)

test_batches = datagen.flow_from_directory(path + 'test/', target_size = (100, 100),
            class_mode = 'categorical', shuffle = True, batch_size = 1)

model.fit_generator(trn_batches, steps_per_epoch = trn_batches.n / 40, epochs = 10, validation_data = test_batches, 
                    validation_steps = test_batches.n / 40)

model.save_weights("model.h5")

model.summary()
