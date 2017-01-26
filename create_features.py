from resize import resize
from csvToDict import csvToDict
from PIL import Image
import numpy as np
import os, csv, random, pickle

def sample_handling(img_dir, classification):

    featureset = []
    i = 1
    for fi in os.listdir(img_dir):
        print(img_dir + ' --- ' + str(i) + ' --- ' + fi)
        # Convert to grayscale
        img = Image.open(os.path.join(img_dir, fi)).convert('L')
        # Resize
        img = resize(img, (100,100), False, False)
        # Convert image to array
        img = np.asarray(img)
        img = img.flatten()
        i += 1

        featureset.append([img,classification])

    return featureset



def create_feature_sets_and_labels(test_size = 0.1):
    features = []
    features += sample_handling('../imgs/train/c0',[1,0,0,0,0,0,0,0,0,0])
    features += sample_handling('../imgs/train/c1',[0,1,0,0,0,0,0,0,0,0])
    features += sample_handling('../imgs/train/c2',[0,0,1,0,0,0,0,0,0,0])
    features += sample_handling('../imgs/train/c3',[0,0,0,1,0,0,0,0,0,0])
    features += sample_handling('../imgs/train/c4',[0,0,0,0,1,0,0,0,0,0])
    features += sample_handling('../imgs/train/c5',[0,0,0,0,0,1,0,0,0,0])
    features += sample_handling('../imgs/train/c6',[0,0,0,0,0,0,1,0,0,0])
    features += sample_handling('../imgs/train/c7',[0,0,0,0,0,0,0,1,0,0])
    features += sample_handling('../imgs/train/c8',[0,0,0,0,0,0,0,0,1,0])
    features += sample_handling('../imgs/train/c9',[0,0,0,0,0,0,0,0,0,1])

    print(len(features[0][0]))

    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    print(len(train_x))
    print(len(train_y))
    print(len(test_x))
    print(len(test_y))

    return train_x,train_y,test_x,test_y



train_x,train_y,test_x,test_y = create_feature_sets_and_labels()
# if you want to pickle this data:
with open('feature_set.pickle','wb') as f:
	pickle.dump([train_x,train_y,test_x,test_y],f)
