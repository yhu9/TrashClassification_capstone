import cv2
import numpy as np
import random
import os
import re
import math
import constants
import scipy.misc
import segmentModule as seg
from matplotlib import pyplot as plt
from multiprocessing import Pool

#reads in training image for cnn using pixel data as the training set
#28 x 28 surrounding area of each pixel used for training
#3x3 conv, 7x7 conv
#all training images must be passed when calling nn.py
def cnn_readOneImg(image_dir):
    inputs = []
    img = cv2.imread(image_dir,cv2.IMREAD_COLOR)
    original,markers = seg.getSegments(img,True)
    original,big_markers = seg.reduceSegments(original,markers)
    uniqueMarkers = np.unique(big_markers)
    for uq_mark in uniqueMarkers:
        region_pixels = imageIn[markers == uq_mark]
        resized = cv2.resize(region_pixels,(constants.IMG_SIZE,constants.IMG_SIZE),interpolation = cv2.INTER_CUBIC)
        inputs.append(resized)

    return(inputs)

#normalizes the image according to the constants file
def normalizeImage(original):
    resized = cv2.resize(original,(constants.IMG_SIZE, constants.IMG_SIZE), interpolation = cv2.INTER_CUBIC)
    return resized

#get the batch of segments to process
def getSegmentBatch(n):
    #initialize variablees
    seg_dir = constants.SEG_DIR
    segment_names = os.listdir(seg_dir)
    inputs = []
    labels = []

    #start a random seed for the random number generator
    random.seed(None)
    for i in range(n):
        #get a random segment from list of segments
        rand_int = random.randint(0,len(segment_names) - 1)
        segment_file = segment_names[rand_int]
        label = re.findall("treematter|construction|cardboard|plywood",segment_file)

        #push the found category into the labels list
        if(label[0] == constants.CAT1):
            labels.append(constants.CAT1_ONEHOT)
        elif(label[0] == constants.CAT2):
            labels.append(constants.CAT2_ONEHOT)
        elif(label[0] == constants.CAT3):
            labels.append(constants.CAT3_ONEHOT)
        elif(label[0] == constants.CAT4):
            labels.append(constants.CAT4_ONEHOT)
        else:
            print(label[0])

        #read the image and push it into the inputs list
        full_dir = seg_dir + segment_file
        img = cv2.imread(full_dir,cv2.IMREAD_COLOR)
        normal_img = normalizeImage(img)

        inputs.append(normal_img)

    return np.array(inputs),np.array(labels)

