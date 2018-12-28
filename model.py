import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import  Dense, Lambda, Conv2D, Dropout, MaxPooling2D,  Convolution2D, Cropping2D, Flatten
from scipy import sparse
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.optimizers import Adam

samples = []
images = []
angles = []
images_dir = "./data/IMG/"
correction_factor = 0.15
batch_size=32

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)

samples.pop(0)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def add_image_data(center_image, center_angle, angles, images):
    images.append(center_image)
    angles.append(center_angle)
    #print(center_image.shape)
    images.append(np.fliplr(center_image))
    angles.append(-center_angle)

def load_img(batch_sample):
    #print(batch_sample)
    img = images_dir+batch_sample.split('/')[-1]
    return cv2.imread(img)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #name = load_img(batch_sample)

                center_angle = float(batch_sample[3])

                add_image_data(load_img(batch_sample[0]), center_angle, angles, images)
                add_image_data(load_img(batch_sample[1]), center_angle + correction_factor, angles, images)
                add_image_data(load_img(batch_sample[2]), center_angle - correction_factor, angles, images)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)







#train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format


model = Sequential()

ch, row, col = 3, 160, 320

# pre-processing
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(65, 320, ch),
        output_shape=(65, 320, ch)))    

model.add(Conv2D(24, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(36, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(48, kernel_size=(3, 3), activation='elu', strides=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3),  activation='elu'))
model.add(Conv2D(64, kernel_size=(3, 3),  activation='elu'))

model.add(Dropout(0.5))

model.add(Flatten())


model.add(Dense(100, activation='elu'))
model.add(Dense(50,  activation='elu'))

model.add(Dense(1,  activation='elu'))


# model.add(Conv2D(64, (5, 5), input_shape=(65, 320, ch)))
# model.add(Activation('elu'))
# model.add(Dropout(0.25))
# model.add(Conv2D(128, (5, 5), input_shape=(65, 320, ch)))
# model.add(Activation('elu'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator,samples_per_epoch=int(len(train_samples)/batch_size),
        nb_epoch=4,
        validation_data=validation_generator,
        nb_val_samples=int(len(validation_samples)/batch_size))

model.summary()

model.save('model.h5')


