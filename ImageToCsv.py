# Convert to grayscale
from PIL import Image
import numpy as np
import pandas as pd
import os, csv
from myResize import myResize

# Specify the location and name of the csv file (can be new)
pixelCsv = open('/Users/jb/Desktop/new.csv', 'wb')
writer = csv.writer(pixelCsv)

# Specify the directory of the
directory = 'imgsTest/trainTest/'

# Loop through each category
for i in range(10):
    for filename in os.listdir(directory + str(i)):
        img = Image.open(os.path.join(directory + str(i),  filename)).convert('L')

        # Resize the image
        img = myResize(img, (100,100))

        # Convert to numpy array
        img_as_np = np.asarray(img)
        img_as_np = img_as_np.flatten()

        # Add the Label to the end of the as array
        img_as_np = np.append(img_as_np, i)

        # Append the array to csv
        writer.writerow(img_as_np)

pixelCsv.close()
