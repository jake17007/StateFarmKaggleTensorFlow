import tensorflow as tf
import numpy as np
from resize import resize
from csvToDict import csvToDict
from PIL import Image
import os, csv, random, pickle

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100
hm_epochs = 20

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weights':tf.Variable(tf.random_normal([7500, n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes]))}

def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

saver = tf.train.Saver()

def classify_img(img_filename):
    prediction = neural_network_model(x)

    with tf.Session() as sess:
        saver.restore(sess, "./model.ckpt")
        # Convert to grayscale
        img = Image.open(os.path.join('imgs/test', img_filename)).convert('L')
        # Resize
        img = resize(img, (100,100), False, False)
        # Convert image to array
        img = np.asarray(img)
        img = img.flatten()

        # Classify image
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[img]}),1)))

        # Print result
        if result[0] == 0:
            print('safe driving: ',img_filename)
        elif result[0] == 1:
            print('texting - right: ',img_filename)
        elif result[0] == 2:
            print('talking on the phone - right: ',img_filename)
        elif result[0] == 3:
            print('texting - left: ',img_filename)
        elif result[0] == 4:
            print('talking on the phone - left: ',img_filename)
        elif result[0] == 5:
            print('operating the radio: ',img_filename)
        elif result[0] == 6:
            print('drinking: ',img_filename)
        elif result[0] == 7:
            print('reaching behind: ',img_filename)
        elif result[0] == 8:
            print('hair and makeup: ',img_filename)
        elif result[0] == 9:
            print('talking to passenger: ',img_filename)

classify_img('img_1.jpg')
print('1.  operating the radio')
classify_img('img_2.jpg')
print('2.  talking on the phone - right')
classify_img('img_3.jpg')
print('3.  talking on the phone - left')
classify_img('img_4.jpg')
print('4.  drinking')
classify_img('img_7.jpg')
print('5.  texting - left')
classify_img('img_8.jpg')
print('6.  texting - left')
classify_img('img_9.jpg')
print('7.  operating the radio')
classify_img('img_10.jpg')
print('8.  operating the radio')
classify_img('img_11.jpg')
print('9.  texting - right')
classify_img('img_12.jpg')
print('10.  texting - right')
classify_img('img_13.jpg')
print('11.  talking on the phone - right')
classify_img('img_15.jpg')
print('12.  talking on the phone - left')
classify_img('img_17.jpg')
print('13.  operating the radio')
classify_img('img_18.jpg')
print('14.  drinking')
classify_img('img_20.jpg')
print('15.  operating the radio')
classify_img('img_21.jpg')
print('16.  drinking')
classify_img('img_22.jpg')
print('17.  safe driving')
classify_img('img_23.jpg')
print('18.  safe driving')
classify_img('img_24.jpg')
print('19.  safe driving')
classify_img('img_25.jpg')
print('20.  talking on the phone - right')
