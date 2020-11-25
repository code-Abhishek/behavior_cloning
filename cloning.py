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

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn


####Use this line to understand

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, /
            steps_per_epoch=ceil(len(train_samples)/batch_size), /
            validation_data=validation_generator, /
            validation_steps=ceil(len(validation_samples)/batch_size), /
            epochs=5, verbose=1)


history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, verbose=1)
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
