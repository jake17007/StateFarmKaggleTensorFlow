# For all train images:
#   Convert it to grayscale
#   Resize the image
#   Convert it to an array
#   Append the array to a csv file
#   Append the label to the csv file

from resize import resize
from csvToDict import csvToDict
from PIL import Image
import numpy as np
import os, csv



preprocess_for_imageflow('../imgsTest/train', 'imagesOutfile.csv', '../img_labels.csv', 'labelsOutfile.csv')
