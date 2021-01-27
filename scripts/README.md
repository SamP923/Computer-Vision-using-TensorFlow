# Python Scripts
    
Original scripts:  
    `renamer.py`:       Renames all of the images in a directory (obselete as `resizer.py` does it as well)  
    `resizer.py`:       Resizes images to have longer dimension be 720, scales other  
    `resizer_v2.py`:    Forces images to resize to 640x480 (will stretch or squeeze images)  
    `splitdataset.py`:  Shuffles the dataset then splits it into training and test folders  
    
Adapted from other sources:
    `findTensorNames.py`:  Find the names of input and output nodes   
    `generate_tfrecord.py`: Create training and testing data from images and .csv annotations
    `model_main.py`: Supplemental training script
    `Object_detection_picamera.py`: Script for running model on Raspberry Pi external PiCamera
    `train.py`: Train the detection model
    `xml_to_csv.py`: Convert .xml files to .csv for image annotations
