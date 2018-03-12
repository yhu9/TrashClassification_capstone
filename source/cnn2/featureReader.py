import cv2
import numpy as np
import random
import os
import re
import math
import constants
import scipy.misc
from matplotlib import pyplot as plt

#reads in training image for cnn using pixel data as the training set
#28 x 28 surrounding area of each pixel used for training
#3x3 conv, 7x7 conv
#all training images must be passed when calling nn.py
def cnn_readOneImg(image_dir):

    img = cv2.imread(image_dir,cv2.IMREAD_COLOR)

    return(img)

#function for stitching 4 different images together
#top left = cat1 = 'treematter'
#top right = cat2 = 'plywood'
#bot left = cat3 = 'cardboard'
#bot right = cat4 = 'construction'
def getStitchedImage():

    #initialize variablees
    cat1_dir = constants.cat1_dir
    cat2_dir = constants.cat2_dir
    cat3_dir = constants.cat3_dir
    cat4_dir = constants.cat4_dir
    cat1files = []
    cat2files = []
    cat3files = []
    cat4files = []

    #check if the file directories exist and push all files into their respective categories
    if os.path.exists(cat1_dir) and os.path.exists(cat2_dir) and os.path.exists(cat3_dir) and os.path.exists(cat4_dir):
        for filename in os.listdir(cat1_dir):
            cat1files.append(filename)
        for filename in os.listdir(cat2_dir):
            cat2files.append(filename)
        for filename in os.listdir(cat3_dir):
            cat3files.append(filename)
        for filename in os.listdir(cat4_dir):
            cat4files.append(filename)

    #pick a random file from the list of files for each category and read them in
    random.seed(None)
    a = random.randint(0,len(cat1files) - 1)
    b = random.randint(0,len(cat2files) - 1)
    c = random.randint(0,len(cat3files) - 1)
    d = random.randint(0,len(cat4files) - 1)
    img1 = cv2.imread(cat1_dir + '/' + cat1files[a],cv2.IMREAD_COLOR)
    img2 = cv2.imread(cat2_dir + '/' + cat2files[b],cv2.IMREAD_COLOR)
    img3 = cv2.imread(cat3_dir + '/' + cat3files[c],cv2.IMREAD_COLOR)
    img4 = cv2.imread(cat4_dir + '/' + cat4files[d],cv2.IMREAD_COLOR)

    #create the image by resizing and putting them into their correct positions
    topleft = cv2.resize(img1,(500,500),interpolation = cv2.INTER_CUBIC)
    bottomleft = cv2.resize(img2,(500,500),interpolation = cv2.INTER_CUBIC)
    topright = cv2.resize(img3,(500,500),interpolation = cv2.INTER_CUBIC)
    bottomright = cv2.resize(img4,(500,500),interpolation = cv2.INTER_CUBIC)
    toprow = np.concatenate((topleft,topright),axis = 1)
    bottomrow = np.concatenate((bottomleft,bottomright),axis = 1)
    full_img = np.concatenate((toprow,bottomrow),axis = 0)

    return full_img

def testStitcher():
    for i in range(10):
        full_img = stitchImage()

        rgb = scipy.misc.toimage(full_img)

        cv2.imshow('stiched image',full_img)
        cv2.imwrite('full_img.png',full_img)
        cv2.waitKey(0)

#gets n patches from an image with its respective label
def getImgBatch(img,n):
    inputs = []
    labels = []

    #get the image shape
    w,h,d = img.shape

    for i in range(n):
        #get a random point on the image that is away from the edge
        random.seed(None)
        a = random.randint(28,w - 29)
        b = random.randint(28,h - 29)

        #28x28 box on random point in the image
        box = img[a-14:a+14,b-14:b+14]

        c = random.randint(0,3)
        if(c == 0):
            M0 = cv2.getRotationMatrix2D((28/2,28/2),0,1)
            cv2.warpAffine(box,M0,(w,h))
        elif(c == 1):
            M1 = cv2.getRotationMatrix2D((28/2,28/2),90,1)
            cv2.warpAffine(box,M1,(w,h))
        elif(c == 2):
            M2 = cv2.getRotationMatrix2D((28/2,28/2),180,1)
            cv2.warpAffine(box,M2,(w,h))
        elif(c == 3):
            M3 = cv2.getRotationMatrix2D((28/2,28/2),270,1)
            cv2.warpAffine(box,M3,(w,h))

        if(a < 500 and b < 500):#topleft
            labels.append(constants.CAT1_ONEHOT)
        elif(a < 500 and b >= 500):#bottomleft
            labels.append(constants.CAT3_ONEHOT)
        elif(a >= 500 and b < 500):#topright
            labels.append(constants.CAT2_ONEHOT)
        elif(a >= 500 and b >= 500):#botright
            labels.append(constants.CAT4_ONEHOT)

        inputs.append(box)

    return inputs,labels

