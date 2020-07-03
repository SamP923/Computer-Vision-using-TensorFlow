## Bulk images renamer

# This script renames all the images in a folder to image_(#).jpg

# Usage: Change the path to the folder where you want the images to be renamed.
#        If you want to, change the names of your files in the last line.

import os
path = '/Users/saman/Documents/GitHub/TensorFlow-TrainingData/playdohRaw' # change to whatever the path to your folder is
files = os.listdir(path)

for index, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(path, 'img_' + str(index) + '.jpg'))
