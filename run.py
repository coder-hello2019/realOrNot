import os
import argparse

ap = argparse.ArgumentParser()

args = ap.parse_args()

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
import numpy as np

def load_data(img_dir):
    # grab paths of the light/dark subfolders
    train_imgs_paths = [os.path.join(img_dir, 'train/', img_path) for img_path in os.listdir(os.path.join(img_dir, 'train/'))]
    test_imgs_paths = [os.path.join(img_dir, 'test/', img_path) for img_path in os.listdir(os.path.join(img_dir, 'test/'))]

    train_datagen = ImageDataGenerator(
        # reshapes the images
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    # flow_from_directory allows for additional data creation
    train_generator = train_datagen.flow_from_directory(
        os.path.join(img_dir, 'train/'),
        target_size=(500, 500),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(img_dir, 'test/'),
        target_size=(500, 500),
        batch_size=20,
        class_mode='binary')   

    return train_generator, validation_generator 

try:
     if __name__ == '__main__':
        load_data('/Users/agatasiuchninska/Documents/Coding/Studying/ML practice/realOrNot/img')

except KeyboardInterrupt:
    print('\nUser aborted!')
