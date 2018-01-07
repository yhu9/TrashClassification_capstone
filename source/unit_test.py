import sys
import segmentModule
import extractionModule
import numpy as np
import random
import cv2
import math
import os
import scipy.misc
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
    image, markers = segmentModule.getSegments(output,showFlag)
    features = []
    uniqueMarkers = np.unique(markers)

    #extractionModule.extractFeatures(image,markers,'edge',SHOW)
    extractionModule.extractAllSegments(image,markers,'edge',showFlag)

# Tests getting the edge histogram distribution feature from the largest region from the lena image
####################################################################################################
def test_getSegments(imageFileIn,showFlag):
    output = segmentModule.normalizeImage(imageFileIn)
    image, markers = segmentModule.getSegments(output,showFlag)

def test_saveSegments(img_dir):

    args = os.listdir(img_dir)
    out_dir = 'segments/'
    for f in args:
        full_dir = img_dir + f

        image = segmentModule.normalizeImage(full_dir)
        original, labels = segmentModule.getSegments(image,False)
        segmentModule.saveSegments(original,labels,False,out_dir,f)

def create_histogram():
    image = np.array([2,4,5,6])

    plt.plot(image,color='black')#   show flag
    plt.show()

def stitchImage():
    img1 = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
    img2 = cv2.imread(sys.argv[3],cv2.IMREAD_COLOR)
    img3 = cv2.imread(sys.argv[4],cv2.IMREAD_COLOR)
    img4 = cv2.imread(sys.argv[5],cv2.IMREAD_COLOR)

    topleft = cv2.resize(img1,(250,250),interpolation = cv2.INTER_CUBIC)
    topright = cv2.resize(img2,(250,250),interpolation = cv2.INTER_CUBIC)
    bottomleft = cv2.resize(img3,(250,250),interpolation = cv2.INTER_CUBIC)
    bottomright = cv2.resize(img4,(250,250),interpolation = cv2.INTER_CUBIC)

    toprow = np.concatenate((topleft,topright),axis = 1)
    bottomrow = np.concatenate((bottomleft,bottomright),axis = 1)
    full_img = np.concatenate((toprow,bottomrow),axis = 0)

    rgb = scipy.misc.toimage(full_img)

    cv2.imshow('stiched image',full_img)
    cv2.imwrite('full_img.png',full_img)
    cv2.waitKey(0)

def saveRotations(image_dir):

    args = os.listdir(image_dir)
    for f in args:
        full_dir = image_dir + f

        original = cv2.imread(full_dir,cv2.IMREAD_COLOR)
        rows,cols,depth = original.shape

        M1 = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        M2 = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        M3 = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
        dst1 = cv2.warpAffine(original,M1,(cols,rows))
        dst2 = cv2.warpAffine(original,M1,(cols,rows))
        dst3 = cv2.warpAffine(original,M1,(cols,rows))

        #cv2.imshow('warped',dst)
        dst_dir1 = image_dir + str(90) + "_" + f
        dst_dir2 = image_dir + str(180) + "_" + f
        dst_dir3 = image_dir + str(270) + "_" + f

        cv2.imwrite(dst_dir1,dst1)
        cv2.imwrite(dst_dir2,dst2)
        cv2.imwrite(dst_dir3,dst3)

        #cv2.waitKey(0)


####################################################################################################
def test_playground():
    print("hello world")


#test_playground()
if(sys.argv[1] == 'rotate'):
    saveRotations(sys.argv[2])
elif(sys.argv[1] == 'save'):
    test_saveSegments(sys.argv[2])
elif(sys.argv[1] == 'stitch'):
    stitchImage()
else:
    #test_getSegments(sys.argv[1],True)
    test_extractEdgeDistribution(sys.argv[1],True)

#rotate_image(sys.argv[1])

