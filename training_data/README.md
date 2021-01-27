# training data
Folder contains all of my training and testing data, including raw

Folders:  
    `playdohObjDet`:    Contains custom object detection model
    `playdohRaw`:       Raw images taken of my cat, Playdoh  
    `playdohResized`:   Resized images of those in `playdohRaw` forced one dim 720, scales other  
    `resized`:          Resized images of those in `playdohRaw` forced 640x480
    
Files:  
    `frozen_inference_graph.pb`: Used to generate Lite model from regular TensorFlow model
    `labelmap.pbtxt`:   Labelmap for TensorFlow model
    `labelmap.txt`:     Labelmap for TensorFlow Lite model

