import csv
from scipy import ndimage 

lines = []
with open('./drive_data/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)  #appending all the data from the reader into 
       
car_images = []
steering_angles = []

for line in lines:
    filename = line[0].split('/')[-1]
    current_path = './drive_data/IMG/' + filename
    image = ndimage.imread(current_path)
    car_images.append(image)
    steering_angles.append(float(line[3]))
    
augmented_images = []
augmented_steering_angles = []

import numpy as np
# print('here')
#augmenting images by flipping them from left to right & same for the measurements.
augmented_car_images = np.fliplr(car_images)
augmented_steering_angles = np.array([-1 * x for x in steering_angles])
# print('a_s_r:', type(augmented_steering_angles))


#adding them to the original dataset
car_images.extend(augmented_car_images)
steering_angles.extend(augmented_steering_angles)
# print(len(car_images))
# print(len(steering_angles))

X_train = np.array(car_images)
y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape = (160,320,3)))
model.add(Lambda(lambda x: (x/255.0) - 0.5))
model.add(Convolution2D(6,(5,5), activation=('relu')))
model.add(MaxPooling2D())
model.add(Convolution2D(6,(5,5), activation=('relu')))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
