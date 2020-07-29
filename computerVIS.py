from cv2 import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras


#### TRAIN IMAGE ########
# load a color image in grayscale
folderName = input("enter filename containing training images ")
directory = '/Users/iankim/Desktop/Python_proj/p6-ort/' + folderName
train_images = []
train_labels = []
for fileName in os.listdir(directory):
    if(os.stat(directory +'/'+ fileName).st_size == 0):
        os.remove(directory +'/'+ fileName)
    else: 
        img = cv2.imread(directory +'/'+ fileName,cv2.IMREAD_COLOR)
# change the size of the image in format (width, height)
    
        resize = cv2.resize(img,(300,200))
        cv2.imshow('img',resize)
        cv2.waitKey(100000)
        cv2.destroyAllWindows()
    # for testing purposes, input correct labels of the buildings
        label = input('what is name of this building')
        train_labels.append(label)
# converting to grayscale
        img = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        train_images.append(img)
# numpy array of image pixels with corresponding labels

####  TEST IMAGES  ######
# load a color image in grayscale
folderName = input("enter filename containing testing images ")
directory = '/Users/iankim/Desktop/Python_proj/p6-ort/' + folderName
test_images = []
test_labels = []
for fileName in os.listdir(directory): 
    if(os.stat(directory +'/'+ fileName).st_size == 0):
        os.remove(directory +'/'+ fileName)
    else:
        img = cv2.imread(directory +'/'+ fileName,cv2.IMREAD_COLOR)
    
# change the size of the image in format (width, height)
        resize = cv2.resize(img,(300,200))
        cv2.imshow('img',resize)
        cv2.waitKey(100000)
        cv2.destroyAllWindows()
    # for testing purposes, input correct labels of the buildings
        label = input('what is name of this building')
        train_labels.append(label)
# converting to grayscale
        img = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        test_images.append(img)
# numpy array of image pixels with corresponding labels


train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (300,200)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics = ['accuracy'])

model.fit(train_images, train_labels,epochs = 10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest accuracy:', test_acc)


probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

