import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from scipy import sparse

lines = []
images = []
measurements = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


def add_img_data_to_training_set(img, measurement):
    images.append(img)
    measurements.append(measurement)
    # add flipped image data
    images.append(np.fliplr(img))
    measurements.append(-1 * measurement)


def load_img_data(file):
    img_path = "./data/IMG/" + file
    return cv2.imread(img_path)


# remove the heading
lines.pop(0)

correction_factor = 0.2


for line in lines:
    current_images = []
    current_measurements = []

    # center left right - in this order
    for i in range(1):
        file = line[i].split('/')[-1]
        current_images.append(load_img_data(file))

    measurement = float(line[3])

    current_measurements.append(measurement) # center
    current_measurements.append(measurement + correction_factor) # left
    current_measurements.append(measurement - correction_factor) # right

    for img, measurement in zip(current_images, current_measurements):
        add_img_data_to_training_set(img, measurement)


X_train = np.array(images)
Y_train = np.array(measurements)
print(X_train.shape)
print(Y_train.shape)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Conv2D(32, (3, 3)))
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=4)
model.summary()
model.save('model.h5')
