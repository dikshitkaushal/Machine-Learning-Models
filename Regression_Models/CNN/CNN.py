# -*- coding: utf-8 -*-
"""
Created on Thu May 21 00:51:19 2020

@author: -
"""

# PART 1 Building the CNN Model

#import the libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialise the CNN
classifier=Sequential()

#step 1 Convolution
classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3) , activation='relu'))

#step 2 pooling 
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step 1 Convolution
classifier.add(Conv2D(32,(3,3) , activation='relu'))

#step 2 pooling 
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step 3 flattening
classifier.add(Flatten())

#step 4 fully connected layer
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='relu'))

#compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#PART 2 FITTING THE CNN TO THE IMAGE

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                 'dataset/training_set',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit(
                training_set,
                steps_per_epoch=8000,
                epochs=25,
                validation_data=test_set,
                validation_steps=2000)
