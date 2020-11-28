# Behavioral Cloning Project

Overview
---
The project comprises of using deep neural networks and convolutional neural networks to clone driving behavior by training, validating and testing a model using keras. The output of the images from the simulator will be used to determine the steering angle for the car be moved for each of the next frame of an autonomous vehicle. 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Text Shortcuts:
code line number : cl#

[//]: # (Image References)

[architecture]: ./examples/architecture.png "Architecture of the model"
[centerimg]: ./examples/center.jpg "Center View from Dashboard"
[rightimg]: ./examples/right.jpg "Right View from Dashboard"
[leftimg]: ./examples/left.png "Left View from Dashboard"
[flippedimg]: ./examples/flipped_img.png "Flipped Image"
[runvideogif]: ./examples/ezgif.com-gif-maker.gif "Run Video from simulator using Model"
[model]: https://drive.google.com/file/d/15VslWOT69Ak3xrfasw2LnoCeOvKYtvQc/view?usp=sharing
[final video]: https://drive.google.com/file/d/10b7NNUPyRBRJmzlOnrNiYP3da_ZGnG8d/view?usp=sharing

#### Ever wondered, what it looks like to let a computer drive like you? I did. Here's what it does:






&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; ![alt text][runvideogif] 








#### Let's see how I got my computer to learn that.

#### Project Contents:
My project includes the following files:
* model(cloning_nvidia_generator).py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network (**WARNING!** Large file can be downloaded here - [model])
* run1.mp4 the final video showcasing my behavior cloned autonomous car. (**WARNING!** Large file can be downloaded here - [final video])
* README.md, summarizing the results. *You are reading it*!

#### Want an Out of the box solution? - functional code:
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```


#### Interested in the inner-workings? - Model Architecture and Training Strategy

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (see architecture image below; cl#66-71) 

The model includes RELU layers to introduce nonlinearity (cl#66-71), and the data is normalized in the model using a Keras lambda layer (cl#64).


The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (cl# 80).

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, which helped train the model to keep in the data points 



#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to utilize the benefits of the CNN already developed by NVIDIA that minimizes the squared-mean error between the steering command output of the network and the adjusted steering command output from the driver.

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, hence to improve the driving behavior in these cases, I then tried using the Lenet model which increased the accuaracy but when the model was used to run the autonomous car in the simulator, it did not completely stay on track, letting me wonder to find another network - the NVIDIA one. Repeating the process again, and hypertuning this network yielded in a model, which at the end of the process, was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (cl#63-80) consisted of a convolution neural network with the following layers and layer sizes.

![alt text][architecture]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][centerimg]

To reduce overfitting the model I utilized the left and the right image of the driving behavior from the center image recorded earlier. I tuned the correction parameter (cl#25) which offsets the steering angle taken from the center, and add into the dataset for training. 

To augment the data set further, I also flipped images and angles thinking that this would help further reduce overfitting the model given the prior knowledge of the scenery. For example, here is an image that has then been flipped:


![alt text][centerimg] &emsp; &emsp;  ![alt text][flippedimg]

&emsp; &emsp; &emsp; &emsp; &emsp; Center Image &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; Flipped Image


After the collection process, I preprocessed the data by normalizing it using the lambda function (cl#64) 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.



## Details about model saving and running

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.



### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.


