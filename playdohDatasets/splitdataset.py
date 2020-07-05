# Usage: place this script in a folder of images you want to randomly
#        sort into training and test sets, then run it. It will create
#        two subfolders called training_images and testing_images

from random import shuffle
from math import floor
from glob import glob
import os, shutil
import numpy as np

from sklearn.model_selection import train_test_split

os.mkdir('training_images')
os.mkdir('testing_images')

source_path = os.getcwd()
training_path = source_path + '\\training_images'
testing_path = source_path + '\\testing_images'

files = glob(source_path + "\\*.jpg")
shuffle(files)
dataset_size = len(files)

# change the percentage of the dataset that are training and test sets
train_size = 0.80
test_size = 0.20

# calculates the size of the training and test files
# validation set can be cut out of the training files later
training_files = floor( train_size * dataset_size)
test_files = dataset_size - training_files


training_dataset, test_dataset = train_test_split(files, train_size = training_files, test_size = test_files)
print('Training set: ' + len(training_dataset))
print('Testing set: ' + len(test_dataset))

for file in test_dataset:
    shutil.move(file, testing_path)

for file in training_dataset:
    shutil.move(file, training_path)


