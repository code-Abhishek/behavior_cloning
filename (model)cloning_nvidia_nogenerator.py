import csv
from scipy import ndimage 

lines = []
with open('./drive_data/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)  #appending all the data from the reader into 
       
car_images = []
steering_angles = []

print(len(lines))
for line in lines:
    c_filename = line[0].split('/')[-1]
    l_filename = line[1].split('/')[-1]
    r_filename = line[2].split('/')[-1]
    
    center_path = './drive_data/IMG/' + c_filename
    left_path = './drive_data/IMG/' + l_filename
    right_path = './drive_data/IMG/' + r_filename
    steering_center = float(line[3])

    # create adjusted steering measurements for the side camera images
    correction = 0.15        # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    center_image = ndimage.imread(center_path)
    left_image = ndimage.imread(left_path)
    right_image = ndimage.imread(right_path)
    
    car_images.extend((center_image, left_image, right_image))
    steering_angles.extend((steering_center, steering_left, steering_right))
    
print(len(car_images))
print(len(steering_angles))
augmented_images = []
augmented_steering_angles = []

import numpy as np
#augmenting images by flipping them from left to right & same for the measurements.
augmented_car_images = np.fliplr(car_images)
augmented_steering_angles = np.array([-1 * x for x in steering_angles])



#adding them to the original dataset
car_images.extend(augmented_car_images)
steering_angles.extend(augmented_steering_angles)

print(len(car_images))
print(len(steering_angles))

X_train = np.array(car_images)
y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,(5,5), subsample=(2,2), activation=('relu')))
# model.add(MaxPooling2D())
model.add(Convolution2D(36,(5,5), subsample=(2,2), activation=('relu')))
model.add(Convolution2D(48,(5,5), subsample=(2,2), activation=('relu')))
model.add(Convolution2D(64,(3,3), activation=('relu')))
model.add(Convolution2D(64,(3,3), activation=('relu')))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, verbose=1)
print(history_object.history.keys())

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save('model.h5')
