from random import shuffle
from math import floor
from glob import glob
import os, shutil
import numpy as np

from sklearn.model_selection import train_test_split

files = glob(r"C:\Users\saman\Documents\GitHub\TensorFlow-TrainingData\playdohResized\*.jpg")

shuffle(files)
dataset_size = len(files)

# change the percentage of the dataset that are training and test sets
train_size = 0.80
test_size = 0.20

# set the path of the folders
source_path = r"C:\Users\saman\Documents\GitHub\TensorFlow-TrainingData\playdohResized"
training_path = r"C:\Users\saman\Documents\GitHub\TensorFlow-TrainingData\playdohTrainingImages"
testing_path = r"C:\Users\saman\Documents\GitHub\TensorFlow-TrainingData\playdohTestingImages"

# calculates the size of the training and test files
# validation set can be cut out of the training files later
training_files = floor( train_size * dataset_size)
test_files = dataset_size - training_files


training_dataset, test_dataset = train_test_split(files, train_size = training_files, test_size = test_files)
print(len(training_dataset))
print(len(test_dataset))

for file in test_dataset:
    shutil.move(file, testing_path)

for file in training_dataset:
    shutil.move(file, training_path)


