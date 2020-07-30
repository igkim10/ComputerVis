from cv2 import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras


#### TRAIN IMAGE ########
# load a color image in grayscale
folderName = input("enter filename containing training images ")
directory = '/Users/iankim/Desktop/Python_proj/p6-computervis/' + folderName
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
directory = '/Users/iankim/Desktop/Python_proj/p6-computervis/' + folderName
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



