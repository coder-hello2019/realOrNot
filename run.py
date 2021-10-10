import os
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import cv2
import numpy as np

# also would be well-built as a class
m = Sequential()
im_height, im_width = 150, 150

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
        target_size=(im_height, im_width),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(img_dir, 'test/'),
        target_size=(im_height, im_width),
        batch_size=20,
        class_mode='binary')   

    return train_generator, validation_generator 

def build_model():

     # our img inputs are arrays - we're making the data more friendly to the machine here; in convolution output goes into the input again
    # N.B. don't need to specify input_shape for subsequent layers because keras infers that - we just need to include in our input layer
    m.add(Conv2D(64, (3, 3), input_shape=(im_height, im_width, 3), activation='relu'))

    # looking at maximum values in a given range
    m.add(MaxPooling2D(pool_size=(5,5)))

    # turning array into a single 'stream' of data
    m.add(Flatten())

    # add to prevent overfitting (loss was consistently going down epoch to epoch prior to this)
    m.add(Dropout(0.4))

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
        steps_per_epoch=20,
        epochs=10,
        validation_data = test_data,
        validation_steps=20,
        use_multiprocessing=False,
        # question: if the workers parameter is removed, the model throws a 'not enough data' error. Nothing on this on google/SO. Investigate why this is happenning.
        workers = 4
    )

# check - might be a python version for this (v:k for k, v in d. items)
def invert_mapping(d):
    inverted = dict()
    for key,value in d.items():
        inverted[value] = key
    return inverted

def predict(img_dir):
    # fetch images
    img_paths = [os.path.join(img_dir, 'predict/', img_path) for img_path in os.listdir(os.path.join(img_dir, 'predict/'))]
    
    # cv2 read in and resize images
    images = [cv2.imread(img) for img in img_paths]
    images = [cv2.resize(img, (im_height, im_width)) for img in images]
    # np reshape
    images = [np.reshape(img, [1, im_height, im_width, 3]) for img in images]
    # return the predictions

    return [(m.predict_classes(img)[0][0], img_paths[i]) for i, img in enumerate(images)]

try:
     if __name__ == '__main__':
        train_data, test_data = (load_data('img'))
        print(f"Types of train_data and test_data: {type(train_data)}, {type(test_data)}")
        build_model()
        train_model(train_data, test_data)
        predictions = predict('img')
        mapping = invert_mapping(train_data.class_indices)

        for val, im_name in predictions:
            print(f'We think that {im_name} is {mapping[val]}')

except KeyboardInterrupt:
    print('\nUser aborted!')