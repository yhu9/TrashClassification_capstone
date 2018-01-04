import sys
import segmentModule
import extractionModule
import numpy as np
import random
import cv2
import math
import os
from matplotlib import pyplot as plt

# Tests getting the color histogram distribution feature from the largest region from the lena image
#####################################################################################################
def test_extractColorDistribution(imageFileIn,showFlag):
    output = segmentModule.normalizeImage(imageFileIn)
    image, markers = segmentModule.getSegments(output,'noshow')
    features = []
    uniqueMarkers = np.unique(markers)

    SHOW = (showFlag == 'SHOW')
    extractionModule.extractAllSegments(image,markers,'color',SHOW)

# Tests getting the edge histogram distribution feature from the largest region from the lena image
#####################################################################################################
def test_extractEdgeDistribution(imageFileIn,showFlag):
    output = segmentModule.normalizeImage(imageFileIn)
    image, markers = segmentModule.getSegments(output,'noshow')
    features = []
    uniqueMarkers = np.unique(markers)

    SHOW = (showFlag == 'SHOW')
    #extractionModule.extractFeatures(image,markers,'edge',SHOW)
    extractionModule.extractAllSegments(image,markers,'edge',SHOW)

# Tests getting the edge histogram distribution feature from the largest region from the lena image
####################################################################################################
def test_getSegments(imageFileIn,showFlag):
    output = segmentModule.normalizeImage(imageFileIn)
    image, markers = segmentModule.getSegments(output,showFlag)

def test_saveSegments(img_dir,category):

    args = os.listdir(img_dir)
    for f in args:
        full_dir = img_dir + f

    out_dir = 'segments/'

    image = segmentModule.normalizeImage("lena3.jpg")
    original, labels = segmentModule.getSegments(image,False)
    segmentModule.saveSegments(original,labels,False,out_dir,category)

def create_histogram():
    image = np.array([2,4,5,6])

    plt.plot(image,color='black')#   show flag
    plt.show()

####################################################################################################
def test_playground():
    img1 = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
    img2 = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
    img3 = cv2.imread(sys.argv[3],cv2.IMREAD_COLOR)
    img4 = cv2.imread(sys.argv[4],cv2.IMREAD_COLOR)

    topleft = cv2.resize(img1,(500,500),interpolation = cv2.INTER_CUBIC)
    topright= cv2.resize(img2,(500,500),interpolation = cv2.INTER_CUBIC)
    bottomleft = cv2.resize(img3,(500,500),interpolation = cv2.INTER_CUBIC)
    bottomright = cv2.resize(img4,(500,500),interpolation = cv2.INTER_CUBIC)

    toprow = np.concatenate((topleft,topright),axis = 1)
    bottomrow = np.concatenate((bottomleft,bottomright),axis = 1)
    full_img = np.concatenate((toprow,bottomrow),axis = 0)

    img = cv2.cvtColor(full_img,cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()

#test_playground()
test_saveSegments(sys.argv[1],sys.argv[2])

