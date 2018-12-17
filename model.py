import csv
import numpy as np
import matplotlib.image as mpimg
import cv2

lines = []

# Open the csv log file and read the file name of the figure
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
    
for line in lines:
    local_path = line[0]
    image = cv2.imread(local_path)
    images.append(image)
    correction = 0.2 
    measurement = float(line[3])
    measurements.append(measurement)
    
    local_path = line[1]
    image = cv2.imread(local_path)
    images.append(image)
    correction = 0.2 
    measurement = float(line[3])
    measurements.append(measurement+correction)
    
    local_path = line[2]
    image = cv2.imread(local_path)
    images.append(image)
    correction = 0.2 
    measurement = float(line[3])
    measurements.append(measurement-correction)

augmented_images = []
augmented_measurements = []

# flip the images to generate more images
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)
    
    
print(len(augmented_measurements))


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def normalizeAndCropLayers():
     """
     Initial pre-processing layers performing normalization and image croping
     """
     model = Sequential()
     model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
     model.add(Cropping2D(cropping=((50,20), (0,0))))
     return model


def modelNvidia():
    """
    Creates nVidia End to End Self-driving Car CNN
    """
    model = normalizeAndCropLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

print('Creating model')
model = modelNvidia()
print('Model ready')
print('Start training the model using MSE loss and Adam optimizer for 5 epochs')
model.compile(optimizer='adam', loss = 'mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=128, nb_epoch = 5)
model.save('model.h5')
print('Model Saved!')

