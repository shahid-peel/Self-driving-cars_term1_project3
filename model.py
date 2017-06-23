import csv
import cv2
import numpy as np
from random import shuffle
import sklearn

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			measurements = []
			# angles = []
			for batch_sample in batch_samples:
				# name = './IMG/'+batch_sample[0].split('/')[-1]
				# center_image = cv2.imread(name)
				# center_angle = float(batch_sample[3])
				# images.append(center_image)
				# angles.append(center_angle)

				source_path = batch_sample[0]
				filename = source_path.split('/')[-1]
				current_path = './data/IMG/' + filename
				image = cv2.imread(current_path)
				#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				images.append(image)
				measurement = float(batch_sample[3])
				measurements.append(measurement)

				source_path = batch_sample[1]
				filename = source_path.split('/')[-1]
				current_path = './data/IMG/' + filename
				image = cv2.imread(current_path)
				# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				images.append(image)
				measurement = float(batch_sample[3]) + 0.20
				measurements.append(measurement)

				source_path = batch_sample[2]
				filename = source_path.split('/')[-1]
				current_path = './data/IMG/' + filename
				image = cv2.imread(current_path)
				# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				images.append(image)
				measurement = float(batch_sample[3]) - 0.20
				measurements.append(measurement)

			aug_images = []
			aug_measurements = []
			for image, measurement in zip(images, measurements):
					image_small = image #cv2.resize(image, (0,0), fx=0.5, fy=0.5)
					aug_images.append(image_small)
					aug_measurements.append(measurement)
					aug_images.append(cv2.flip(image_small,1))
					aug_measurements.append(-measurement)

			# trim image to only see section with road
			X_train = np.array(aug_images)
			y_train = np.array(aug_measurements)
			# y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		try:
			float(line[3])
		except ValueError:
			print("Skipping header line")
			continue
		lines.append(line)

# images = []
# measurements = []
# for line in lines:
# 	try:
# 		float(line[3])
# 	except ValueError:
# 		print("Skipping header line")
# 		continue
#
# 	source_path = line[0]
# 	filename = source_path.split('/')[-1]
# 	current_path = './data/IMG/' + filename
# 	image = cv2.imread(current_path)
# 	#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	images.append(image)
# 	measurement = float(line[3])
# 	measurements.append(measurement)
# 	images.append(np.fliplr(image))
# 	measurements.append(-measurement)
#
#
# X_train = np.array(images)
# y_train = np.array(measurements)

#shuffle(lines)
#shuffle(lines)
#shuffle(lines)
lines = lines[:int(len(lines)*1.0)]
num_of_samples = len(lines)
train_samples = lines[:int(num_of_samples*0.8)]
validation_samples = lines[int(num_of_samples*0.8):]

print('total samples      = ' + str(6*len(lines)))
print('training samples   = ' + str(6*len(train_samples)))
print('validation samples = ' + str(6*len(validation_samples)))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=100)
validation_generator = generator(validation_samples, batch_size=100)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(36, 5, 5, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(48, 5, 5, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Conv2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
# model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
# model.add(Conv2D(48, 5, 5, activation='relu'))
# model.add(Conv2D(64, 3, 3, activation='relu'))
# model.add(Conv2D(64, 3, 3, activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, 3, 3, activation='relu'))

model.add(Flatten())
#model.add(Dense(1000))
# model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

history_object = model.fit_generator(train_generator, samples_per_epoch =
    6*len(train_samples), validation_data =
    validation_generator,
    nb_val_samples = 6*len(validation_samples),
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

