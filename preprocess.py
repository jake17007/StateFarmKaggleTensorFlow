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



# For imageflow
def preprocess_for_imageflow(imagesInDir, imagesOutfile, labelsInfile, labelsOutfile):
    imagesOutfile = open(imagesOutfile, 'w')
    labelsOutfile = open(labelsOutfile, 'w')
    imagesWriter = csv.writer(imagesOutfile)
    labelsWriter = csv.writer(labelsOutfile)
    # Create a dictionary from the filenames (keys) to the labels (values)
    labelsDict = csvToDict(labelsInfile)
    i = 0
    for subdir, dirs, files in os.walk(imagesInDir):
        for f in files:
            if not f.startswith('.'):
                # If you want to save the new images uncomment this line and add
                # 'writeFile' as the last argument in the call to resize()
                # writeFile = open(f+'_new', 'a')

                # Convert to grayscale
                img = Image.open(os.path.join(subdir, f)).convert('L')
                # Resize
                img = resize(img, (100,100), False, False)
                # Convert image to array
                img = np.asarray(img)
                img = img.flatten()
                print('Length of array: ' + str(len(img)))
                # Append image array to given imagesOutfile CSV file
                imagesWriter.writerow(img)
                # Append the label to the csv file
                label = labelsDict[f]
                labelsWriter.writerow(label)
                print(i)
                i += 1

    imagesOutfile.close()
    labelsOutfile.close()

# For reading from csv
def preprocess_for_csvreader(imagesInDir, labelsInfile, outfile):
    outfile = open(outfile, 'w')

    # Create a dictionary from the filenames (keys) to the labels (values)
    labelsDict = csvToDict(labelsInfile)

    for subdir, dirs, files in os.walk(imagesInDir):
        for f in files:
            if not f.startswith('.'):
                # If you want to save the new images uncomment this line and add
                # 'writeFile' as the last argument in the call to resize()
                # writeFile = open(f+'_new', 'a')

                # Convert to grayscale
                img = Image.open(os.path.join(subdir, f)).convert('L')
                # Resize
                img = resize(img, (100,100), False, False)
                # Convert image to array
                img = np.asarray(img)
                img = img.flatten()
                # Append image array to given imagesOutfile CSV file
                imagesWriter.writerow(img)
                # Append the label to the csv file
                label = labelsDict[f]
                labelsWriter.writerow(label)

    imagesOutfile.close()
    labelsOutfile.close()
