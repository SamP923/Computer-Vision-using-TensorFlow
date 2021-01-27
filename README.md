# Computer Vision using TensorFlow
Currently, I'm working on using TensorFlow's Object Detection API on the Raspberry Pi.

## Object Detection with Raspberry Pi using TensorFlow 2.x
*Last updated 7/8/2020*

This tutorial is adapted from [EdjeElectronic’s Tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi) to set up the TensorFlow Object Detection API on the Raspberry Pi. If you run into a bug that’s not covered in this guide, other debugging may be found there. If you've already gone through Edje's guide, skip to the [compatibility](https://github.com/SamP923/custom-object-classification-using-tensorflow#switching-from-tensorflow-1x-to-2x) section.

This guide is compatible with TensorFlow version 2.2.0.

1. Set Up and Update the Raspberry Pi
2. Install TensorFlow and its dependencies
3. Install OpenCV and its dependencies
4. Compile and Install Protobuf
5. Set up TF Directory Structure and PYTHONPATH Environment Variable
6. Download the SSD MobileNet model
7. Download the Object_detection_picamera.py file
8. Detect objects!

Appendix  
- Switching from TF 1.x to 2.x  
- Errors  

### 1. Set Up and Update the Raspberry Pi
If you haven’t already, set up the Raspberry Pi (Raspbian/OS, WiFi, etc.). Whether you prefer using the shell on the GUI or just the shell, either way is fine until you actually run the object detector script, where you will need to use the GUI.  

Update the Raspberry Pi. In a terminal window, issue
```
sudo apt-get update
sudo apt-get dist-upgrade
```
It would also be a good idea to update python and pip here. Any commands later issued from these programs should be python3 and pip3, as this will use Python3 rather than the default Python2

### 2. Install TensorFlow and its dependencies.
TensorFlow needs the LibAtlas package. Install it by issuing
```
sudo apt-get install libatlas-base-dev
```

Pip cannot install versions past TensorFlow 1.15, so we’ll need to build TensorFlow from source to use version 2.x. Pi uses the ARM architecture, so TF needs to be compiled for that specific architecture. Since there isn’t an official release for 2.x, we’ll use [lhelontra’s TensorFlow-on-ARM](https://github.com/lhelontra/tensorflow-on-arm/releases) 2.2.0 release for Pi3 and Debian Buster (3.7). Issue:
```
sudo pip3 install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.2.0/tensorflow-2.2.0-cp37-none-linux_armv7l.whl   
```

The TensorFlow Object Detection API needs a few other things to work. Issue
```
sudo pip3 install pillow lxml jupyter matplotlib cython
sudo apt-get install python-tk
pip3 install git+https://github.com/google-research/tf-slim.git
```

### 3. Install OpenCV and its dependencies
OpenCV has a few more dependencies that it needs in order to work on the RPi

```
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install qt4-dev-tools libatlas-base-dev
```

Now, we can install OpenCV. Issue
```
sudo pip3 install opencv-python==4.1.1.25
```

The version is specified because it seems that some newer versions of OpenCV are [not supported on the Pi](https://github.com/piwheels/packages/issues/59).  

### 4. Compile and Install Protobuf
TensorFlow’s object detection API uses Protobuf, a package that implements Google’s Protocol Buffer data format. To install it, issue
```
sudo apt-get install protobuf-compiler
```

Verify that it’s installed by running `protoc --version`. The output should be something like `libprotoc x.x.x`

### 5. Set up TF Directory Structure and PYTHONPATH Environment Variable
We’ll now set up the TensorFlow directory (where we’ll keep the TF files) and clone the models repo from GitHub. Make the directory, and move into it
```
mkdir tensorflow1
cd tensorflow1
```

Download the TensorFlow models repo from GitHub to your folder by issuing
```
git clone --depth 1 https://github.com/tensorflow/models.git
```

Next, we need to modify the PYTHONPATH environment variable to point at some directories inside the TensorFlow repository we just downloaded. We want PYTHONPATH to be set every time we open a terminal, so we have to modify the .bashrc file. Open it by issuing:
```
sudo nano ~/.bashrc
```
*Note: I actually used Vim as an editor but Edje’s tutorial used nano so I’ll put that here. Use whatever you're comfortable with.*  

Then, move to the very end of the file and add this last line
```
export PYTHONPATH=$PYTHONPATH:/home/pi/tensorflow1/models/research:/home/pi/tensorflow1/models/research/slim
```
Save and exit the file. Close and then re-open the terminal if in GUI, or just reboot it and SSH in again if using just shell.  

We'll use Protoc to compile Protocol Buffer (.proto) files. The proto.files are located in `/research/object_detection/protos`, but we need to execute the command from the /research directory. Issue
```
cd /home/pi/tensorflow1/models/research
protoc object_detection/protos/*.proto --python_out=.
```
This converts all the “name”.proto files to “name_pb2”.py  

Then, move back into the object_detection directory for the next step.

### 6. Download the SSD MobileNet model
There are a variety of pre-trained object detection models created by Google/TensorFlow that you can download and try. Since Pi has a weak processor (compared to a regular computer), we’ll need to use a model that takes less processing power.  

You can check what models are compatible with what versions of TensorFlow by going into /object_detection/builders and reading model_builder.py. Check for models under the if tf_version.is_tf2(): statement
For demo purposes, I used the first model, ssd_mobilenet_v1_coco.tar.gz. You can find all of the models in TensorFlow's model zoo [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

Download the model and unpack it by issuing 
```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xzvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

After you untar the file, it should output the files that are in the new directory. Check if it has a file called `frozen_inference_graph.pb`. If it does, skip to the next step! If it does, oh well, continue on.

#### Create the `frozen_interference_graph.pb` file
We need to create the `frozen_interference_graph.pb` file in order for the `object_detection_picamera.py` file in the next step to work. To do this, we’ll run `export_inference_graph.py` with a few arguments.

Example usage (\ means that this should all be on one line)
```
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path path/to/model.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory
```
Example with coco model
```
python3 export_inference_graph.py \ 
    --input_type image_tensor --pipeline_config_path ./samples/configs/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix ./ssd_mobilenet_v1_coco_2018_01_28/model.ckpt.data-00000-of-00001 \ 
    --output_directory ships_inference_graph
```

Continue to the next step. If, when running `Object_detection_picamera.py` it still can’t find the frozen inference graph, you may be able to [manually create a checkpoint](https://www.tensorflow.org/guide/checkpoint) to run the `export_inference_graph.py` script. However, you should first check that your `--trained_checkpoint_prefix` above actually points to a file that exists in your model directory. If it’s a custom model, it likely will look like `model.ckpt-XXXX`. Use the highest value checkpoint, as that will be the most recent version of the model. If you’re using a model out-of-the-box from the TF model zoo, you could probably just use `model.ckpt.data-00000-of-00001`.

### 7. Download the Object_detection_picamera.py file
Download the `Object_detection_picamera.py` file from EdjeElectronics and make some changes to make it compatible with TensorFlow 2.x. Check out the TF Symbols Map for converting commands between TF1 and TF2 [here](https://docs.google.com/spreadsheets/d/1FLFJLzg7WNP6JHODX5q8BDgptKafq_slHpnHVbJIteQ/edit#gid=0).  

Edje’s script can be downloaded using
```
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/master/Object_detection_picamera.py
```

There is also a reference copy with the following changes in this repo.
  - Edit the file and change the variable MODEL_NAME to whatever model you’re using.
  - Change `od_graph_def = tf.GraphDef()` to `od_graph_def = tf.compat.v1.GraphDef()`
  - Change `sess = tf.Session(graph=detection_graph)` to `sess = tf.compat.v1.Session(graph=detection_graph)`
  - Change `tf.gfile.GFile()` to `tf.compat.v2.io.gfile.GFile()`

### 8. Detect Objects!
If using the picamera, you’ll need to enable it before we run the script.  

GUI: RaspberryPi Symbol>Preferences>Raspberry Pi Configuration>Interfaces>Camera ⇒ enable  
Shell: issue sudo raspi-config>Interfacing Options>Camera>Enable  

Run the script by issuing
```
python3 Object_detection_picamera.py
```
If you’re using a USB-webcam, add --usbcam to the end of the command (I haven’t personally tested this, so there may be some additional things you need to do; ie. edit to make compatible with TF2, get some files from Edje's original repo)
```
python3 Object_detection_picamera.py --usbcam
```

## Appendix
### Switching from TensorFlow 1.x to 2.x
#### Install 2.x
If you’ve already used Pip to install TF version 1.x, you’ll need to uninstall it both locally and globally on the Pi before you can use 2.x, as the 1.x install will override the 2.x. Otherwise, this might cause a Hadoop error when importing TF.  

Then, issue
```
sudo pip3 install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.2.0/tensorflow-2.2.0-cp37-none-linux_armv7l.whl
```  

You can test if it was installed correctly by issuing `python3`, then `import tensorflow`. If no errors appear, you’re good to go!

#### Reinstalling OpenCV  
If it cannot find module cv2, you’ll need to install a lower version of OpenCV. This seems to be because some newer versions of OpenCV are [not supported on the Pi](https://github.com/piwheels/packages/issues/59). I used version 4.1.1.25, but others may work. Make sure that no higher version is installed locally (no sudo when issuing) or globally (sudo when issuing). You can check what version of opencv you have installed by issuing `opencv-python --version`. If you want to see all of the packages you have installed and their versions, issue `pip3 list`. 
```
sudo pip3 uninstall opencv-python
sudo pip3 install opencv-python==4.1.1.25
```

You may need to delete the actual cv2 folder (`rm -r /home/rest-of-path`) for this to work.
```
Sample: rm -r /home/pi/.local/lib/python3.7/site-packages/cv2/
```

You can test if it was installed correctly by issuing `python3`, then `import cv2`.  

#### Edit the Object_detection_picamera.py file
If you've used Edje's tutorial, there are some edits that need to be made to the `Object_detection_picamera.py` file in order for it to be compatible with TensorFlow 2.x. Check out the TF Symbols Map for converting commands between TF1 and TF2 [here](https://docs.google.com/spreadsheets/d/1FLFJLzg7WNP6JHODX5q8BDgptKafq_slHpnHVbJIteQ/edit#gid=0). 

There is also a reference copy with the following changes in this repo.
  - Edit the file and change the variable MODEL_NAME to whatever model you’re using.
  - Change `od_graph_def = tf.GraphDef()` to `od_graph_def = tf.compat.v1.GraphDef()`
  - Change `sess = tf.Session(graph=detection_graph)` to `sess = tf.compat.v1.Session(graph=detection_graph)`
  - Change `tf.gfile.GFile()` to `tf.compat.v2.io.gfile.GFile()`
  
### Errors
#### Stuck on GRPCIO when installing TensorFlow
If your Pi gets stuck on grpcio when installing TF, cancel the install. We can directly install the compiled wheels by issuing 
```
pip3 install https://www.piwheels.org/simple/grpcio/grpcio-1.30.0rc1-cp37-cp37m-linux_armv7l.whl#sha256=76355258b23889570881a18a7a5ed4e303b1e5e5fe03a521f95d2ff2f5449cf4. 
```
Then, try installing TF again. You can find other versions of grpcio [here](https://www.piwheels.org/project/grpcio/).

#### Tensorboard version error
If you get a tensorboard version error, try issuing `pip3 install setuptools --upgrade` to update it.  

#### Memory Allocation Warning when Running Object_detection_picamera.py script
You may need to decrease the batch size in the model `.config file` if TensorFlow gives you a memory allocation warning (it may also run the object-detector later anyway)  

Find .config files in `models/research/object_detection/samples/configs`, then choose the one according to what model you’re using, then change the batch size.


## Legacy Content
This repo previously hosted a clone of Tensorflow's examples repository. I am now working with TensorFlow's models repository. I had been working on retraining the image classification model to recognize flowers.

### Helpful Links:
[Original Repository](https://github.com/tensorflow/examples "TensorFlow's example repository")  
[TensorFlow's overview of the image classifier](https://www.tensorflow.org/lite/models/image_classification/overview "TensorFlow Image Classification")  
[Colab for image classification](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/#0 "Recognize Flowers with TensorFlow on Android")  
Two options to recognize flowers with TensorFlow Lite:  
    [Colab for TFLite model customization](https://colab.research.google.com/github/tensorflow/examples/blob/master/tensorflow_examples/lite/model_customization/demo/image_classification.ipynb "Image Classification")  
    [Colab for transfer learning](https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/flowers_tf_lite.ipynb "Flowers TF Lite")

[Other TensorFlow models](https://www.tensorflow.org/lite/models "TensorFlow Lite Models") <br>
[25 Open Datasets for Deep Learning](https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/ "25 Open Datasets for Deep Learning")
