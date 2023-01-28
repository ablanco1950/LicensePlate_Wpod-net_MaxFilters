# LicensePlate_Wpod-net_MaxFilters
It's a Wpod-net demo, downloaded from https://github.com/quangnhat185/Plate_detect_and_recognize,   for the recognition of car license plates, the use of labeled images is avoided, although the precision is lower. Testing a sample of 21 images of Spanish license plates (NNNNAAA format), 16 hits are obtained. While with https://github.com/ablanco1950/LicensePlate_Labeled_MaxFilters that uses the same images, but labeled, and also an exhaustive number of filters,  17 hits are reached

Requirements:

Download all the files to a directory and unzip the folder with the images to test: test6Training\images ( from roboflow without labels directory)

There must be installed the modules that allow:

import cv2

import numpy as np

from os.path import splitext, basename

from keras.models import model_from_json

import os

import re

import pytesseract

import imutils

import tensorflow (it appears that tensorflow only works with 64-bit)

Execute:

GetNumberSpanishLicensePlate_Wpod-net_MaxFilters.py

Note:
the module local_utils.py, downloaded from https://github.com/quangnhat185/Plate_detect_and_recognize has been  retouched  on line 175 to avoid termination
in case of car license plate not detected. In this manner, the recognition of license is forced increasin automatically the parameter Dmin, as is explained in https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-1-detection-795fda47e922
"If there is no plate founded, the program would warn you with an error “No License plate is founded!”. In this case, try to increase Dmin value to adjust the boundary dimension."

pytesseract is used as OCR

As output, the LicenseResults.txt file is also obtained with the relation between true license plate and predicted license plate.

References:

 https://github.com/quangnhat185/Plate_detect_and_recognize
 
 https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-1-detection-795fda47e922
 
 https://github.com/ablanco1950/LicensePlate_Labeled_MaxFilters
 
 https://www.roboflow.com
 
 https://ksingh7.medium.com/part-i-using-a-pre-trained-keras-model-for-license-plate-detection-7687739af758

