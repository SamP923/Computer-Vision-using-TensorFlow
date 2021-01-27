## Bulk image resizer
# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2, os
from PIL import Image, ExifTags
import traceback, functools
from math import floor

dir_path = os.getcwd() + "\playdohRaw"
print(dir_path)
class_name = 'PLAYDOH'
os.mkdir('resized')
         
for filename in os.listdir(dir_path):
    # If the images are not .JPG images, change the line below to match the image type.
    if filename.endswith(".jpg"):
        img = Image.open(filename)
        
        width, height = img.size
        if width > height: 
            basewidth = 720
            new_height = floor(basewidth * height / width)

            img = img.resize((basewidth, new_height), Image.ANTIALIAS)
        else:
            baseheight = 720
            new_width = floor(baseheight * width / height)
            img = img.resize((new_width, baseheight), Image.ANTIALIAS)

        img.save('resized/' + class_name + '_' + str(idx+1).zfill(3) + '.jpg')


            

        
        
       
