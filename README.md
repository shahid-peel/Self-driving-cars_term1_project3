# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on `Nvidia end-to-end learning` convolutional neural network. It consists of 5 convolutional layers and with relu activation function. First three layers have 5x5 filter with 2x2 sub-sampling whereas the last 2 have 3x3 filters with no sub-sampling.

The relu layers introduce nonlinearity in the network and the data is normalized in the model using a Keras lambda layer.  Apart from this I crop the images after normalization to get rid of parts of image are not relevant to our training.

After the conv layers come the fully connected layers and they gradually reduce the output size with the last layer having only 1 output (steering angle).

#### 2. Attempts to reduce overfitting in the model

I had introduced drop out layers in my initial models but they were not working well for me and the car kept getting stuck (would swerve off) in the turn after the bridge.  The final implementation does not contain any dropout layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting.  I used a 80-20 split for the training. The model was tested by running it through the simulator and ensuring that the vehicle stayed on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the following to create my data:
- used left, center and right camera images
- added steering angle correction for left and right camera images
- flipped the images horizontally so that there is no bias is turning left

Initially I had played around with grayscale images as well but it didn't perform well.

I did however add more data to my training as I saw that the car was getting stuck at certain points on the track. I did around 10 training runs for certain curves in the track to help the model train well for the curves.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have a lot of drop out layers as that was a clever trick that resulted in improvements on the imagenet database. That however didn't work well for me and for the longest time I could not make progress. I felt that there are 3 factors that can have an effect on the accuracy of the trained model:
- good data: in our case, using images from 3 camers and flipping images. Also if the model is making a wrong decision at a particular point in track, then help it by augmenting more training data for those parts of track.
- good model: one that works well with the images and can train well (this seems more like an art at this point)
- suitable hyperparameters: parameters like epochs, batch size etc


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Generally my training error was half the validation error. But both kept decreasing with the number of epochs.  At 4-5 epochs I reached the best result with my model. Instead of training more, i fixed the number of epochs to these numbers so that I don't overfit.

The final step was to run the simulator to see how well the car was driving around track one. When the model had trained well, I could feel that it sayed closer to the center of lane, or perhaps even slightly on the left side of road. My earlier failed models would slowly drift over time towards the right side curb.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

```pythong
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. I my initial attempts I was using the mouse and got very smooth training runs on the track. Later on I shift to using the mouse as my steering angles had move variation in that case.

For a very long time my car would not get past the the left turn immediately after the bridge.  I recorded the vehicle recovering from the right sides of that turn back to center so that the vehicle would learn to correct the steering angle for the turn and not head straight.

After the collection process, I had the following data points:

`total samples      = 34944
training samples   = 27954
validation samples = 6990`

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4-5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Overall a tricky project. It was project #3 but because I wasn't making progress on it, I moved on to #4 and #5 and in the end came back to proejct 3 to finish it off. I think being able to visualize the network at different stages would help so that we can see how to model is learning and get a better intuition about changing parameters.
