# TensorFlow-TrainingData
repo hosts all of my training and testing data

Folders:  
    `playdohRaw`:       Raw images taken of my cat, Playdoh  
    `playdohResized`:   Resized images of those in `playdohRaw` forced one dim 720, scales other  
    `resized`:          Resized images of those in `playdohRaw` forced 640x480
    
Files:  
    `renamer.py`:       Renames all of the images in a directory (obselete as `resizer.py` does it as well)  
    `resizer.py`:       Resizes images to have longer dimension be 720, scales other  
    `resizer_v2.py`:    Forces images to resize to 640x480 (will stretch or squeeze images)  
    `splitdataset.py`:  Shuffles the dataset then splits it into training and test folders  
    
