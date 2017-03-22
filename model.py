#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Flatten, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

def random_bright(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_brightness = .25 + np.random.uniform()
    hsv_image[:,:,2] = hsv_image[:,:,2] * random_brightness
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return rgb_image

def resize(image):
    return cv2.resize(image, (64, 64))

def crop(image):
    return image[55:135, :, :]

def normalize(image):
    image = image.astype(np.float32)
    return image / 255 - 0.5

def flip(image):
    return cv2.flip(image, 1)
    
def preprocess_image(image):
    image = crop(image)
    image = resize(image)
    image = normalize(image)
    return image

def random_choose_direction(data):
    steering = data['steering']
    camera = np.random.choice(['left', 'center', 'right'])
    if camera == 'left':
        steering += 0.229
    elif camera == 'right':
        steering -= 0.229
    image = load_img('udacity_data/' + data[camera].strip())
    image = img_to_array(image)
    return image, steering
    
def random_flip(image, steering):
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        steering *= -1
        image = flip(image)
    return image, steering

def random_process_image(data):
    image, steering = random_choose_direction(data)
    image, steering = random_flip(image, steering)
    image = random_bright(image)
    image = preprocess_image(image)
    return image, steering        

def image_generator(dataset, batch_size=32):
    dataset_count = dataset.shape[0]
    batches = dataset_count // batch_size
    
    batch_num = 0
    while True:
        start = batch_num * batch_size
        end = (batch_num+1) * batch_size -1
        
        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size, ), dtype=np.float32)
        
        i = 0
        for index, data in dataset.loc[start:end].iterrows():
            X_batch[i], y_batch[i] = random_process_image(data)
            i += 1
        
        batch_num += 1

        if batch_num == batches - 1:
            batch_num = 0
        
        yield X_batch, y_batch
        
def get_model():
    
    learning_rate = 0.00001
    activation_relu = 'relu'
    
    model = Sequential()
    
    # starts with five convolutional and maxpooling layers
    model.add(Convolution2D(24, 5, 5, input_shape=(64, 64, 3), border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Flatten())
    
    # Next, five fully connected layers
    model.add(Dense(1164))
    model.add(Activation(activation_relu))
    
    model.add(Dense(100))
    model.add(Activation(activation_relu))
    
    model.add(Dense(50))
    model.add(Activation(activation_relu))
    
    model.add(Dense(10))
    model.add(Activation(activation_relu))
    
    model.add(Dense(1))
    
    model.summary()
    
    model.compile(optimizer=Adam(learning_rate), loss="mse", )
    
    return model

if __name__ == '__main__':
    BATCH_SIZE = 32

    data_frame = pd.read_csv('udacity_data/driving_log.csv', usecols=[0, 1, 2, 3])
    
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    training_split = 0.8

    num_rows_training = int(data_frame.shape[0]*training_split)

    training_data = data_frame.loc[0:num_rows_training-1]
    validation_data = data_frame.loc[num_rows_training:]
    
    training_generator = image_generator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = image_generator(validation_data, batch_size=BATCH_SIZE)
    
    model = get_model()

    samples_per_epoch = (20000//BATCH_SIZE)*BATCH_SIZE

    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=10, nb_val_samples=3000)

    print("Saving model weights and configuration file.")
    model.save('model.h5')
    
    
    
    





