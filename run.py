import os
import argparse

ap = argparse.ArgumentParser()

args = ap.parse_args()

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import cv2
import numpy as np

# also would be well-built as a class
m = Sequential()

def load_data(img_dir):
    # grab paths of the light/dark subfolders
    #train_imgs_paths = [os.path.join(img_dir, 'train/', img_path) for img_path in os.listdir(os.path.join(img_dir, 'train/'))]
    #test_imgs_paths = [os.path.join(img_dir, 'test/', img_path) for img_path in os.listdir(os.path.join(img_dir, 'test/'))]

    train_datagen = ImageDataGenerator(
        # reshapes the images
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    # flow_from_directory allows for additional data creation
    train_generator = train_datagen.flow_from_directory(
        os.path.join(img_dir, 'train/'),
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(img_dir, 'test/'),
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')   

    return train_generator, validation_generator 

def build_model():

     # our img inputs are arrays - we're making the data more friendly to the machine here; in convolution output goes into the input again
    # N.B. don't need to specify input_shape for subsequent layers because keras infers that
    m.add(Conv2D(64, (3, 3), input_shape=(150, 150, 3), activation='relu'))

    # looking at maximum values in a given range
    m.add(MaxPooling2D(pool_size=(5,5)))

    # turning array into a single 'stream' of data
    m.add(Flatten())

    # 64 = units i.e. a positive int determining the dimensionality of the output space - this is a hyperparameter that is tuned during the validation stage
    m.add(Dense(32, activation='sigmoid'))

    # we just want 1 output from our model, telling us whether image is real or not
    m.add(Dense(1, activation='sigmoid'))

    m.compile(
        optimizer='adam',
        metrics=['accuracy'],
        loss='binary_crossentropy'
    )

def train_model(train_data, test_data):
    m.fit(
        x=train_data,
        steps_per_epoch=1,
        epochs=1,
        validation_data = test_data,
        validation_steps=20
    )


try:
     if __name__ == '__main__':
        train_data, test_data = (load_data('img'))
        print(f"Types of train_data and test_data: {type(train_data)}, {type(test_data)}")
        build_model()
        train_model(train_data, test_data)

except KeyboardInterrupt:
    print('\nUser aborted!')
