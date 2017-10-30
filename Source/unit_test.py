import sys
import segmentModule
import extractionModule
import numpy as np
import random
import cv2
import math
from matplotlib import pyplot as plt


# Tests getting the color histogram distribution feature from the largest region from the lena image
#####################################################################################################
def test_extractColorDistribution(imageFileIn,showFlag):
    image, markers = segmentModule.getSegments(imageFileIn,showFlag)

    features = []
    uniqueMarkers = np.unique(markers)
    largest_marker = 0;
    largest_count = 0;
    for x in uniqueMarkers:
        count = np.count_nonzero(markers == x)
        if(count > largest_count):
            largest_marker = x
            largest_count = count

    extractionModule.extractColorDistribution(image,largest_marker,markers,showFlag == "SHOW",0)

# Tests getting the edge histogram distribution feature from the largest region from the lena image
#####################################################################################################
def test_extractEdgeDistribution(imageFileIn,showFlag):
    image, markers = segmentModule.getSegments(imageFileIn,showFlag)

    features = []
    uniqueMarkers = np.unique(markers)
    largest_marker = 0;
    largest_count = 0;
    for x in uniqueMarkers:
        count = np.count_nonzero(markers == x)
        if(count > largest_count):
            largest_marker = x
            largest_count = count

    extractionModule.extractEdgeDistribution(image,largest_marker,markers,showFlag == "SHOW",0)

# Tests getting the edge histogram distribution feature from the largest region from the lena image
####################################################################################################
def test_getSegments(imageFileIn,showFlag):
    image, markers = segmentModule.getSegments(imageFileIn,showFlag)

# Tests getting a window slice from a matrix
####################################################################################################
def test_getslice():
    #create images
    diag1 = np.array([[255,0,0,0,0,0,0],[255,255,0,0,0,0,0],[255,255,255,0,0,0,0],[255,255,255,255,0,0,0],[255,255,255,255,255,0,0],[255,255,255,255,255,255,0],[255,255,255,255,255,255,255]],dtype=float)
    diag2 = np.array([[0,0,0,0,0,0,255],[0,0,0,0,0,255,255],[0,0,0,0,255,255,255],[0,0,0,255,255,255,255],[0,0,255,255,255,255,255],[0,255,255,255,255,255,255],[255,255,255,255,255,255,255]],dtype=float)
    vert = np.array([[255,255,255,255,0,0,0],[255,255,255,255,0,0,0],[255,255,255,255,0,0,0],[255,255,255,255,0,0,0],[255,255,255,255,0,0,0],[255,255,255,255,0,0,0],[255,255,255,255,0,0,0]],dtype=float)
    horiz = np.array([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[255,255,255,255,255,255,255],[255,255,255,255,255,255,255],[255,255,255,255,255,255,255],[255,255,255,255,255,255,255]],dtype=float)

    #create the kernals
    #not sure what the 5x5 version of the diagnol is.
    #http://www.cim.mcgill.ca/~image529/TA529/Image529_99/assignments/edge_detection/references/sobel.htm
    K1 = [[1,2,1],[0,0,0],[-1,-2,-1]]         #horizontal
    K2 = [[1,0,-1],[2,0,-2],[1,0,-1]]         #vertical
    K3 = [[2,2,-1],[2,-1,-1],[-1,-1,-1]]      #diagnol1
    K4 = [[-1,2,2],[-1,-1,2],[-1,-1,-1]]      #diagnol2
    K5 = [[1,4,6,4,1],[2,8,12,8,2],[0,0,0,0,0],[-2,-8,-12,-8,-2],[-1,-4,-6,-4,-1]]         #horizontal
    K6 = [[1,2,0,-2,-1],[4,8,0,-8,-4],[6,12,0,-12,-6],[4,8,0,-8,-4],[1,2,0,-2,-1]]         #vertical

    kernal1 = np.array(K1,dtype=float)/8
    kernal2 = np.array(K2,dtype=float)/8
    kernal3 = np.array(K3,dtype=float)/8
    kernal4 = np.array(K4,dtype=float)/8
    kernal5 = np.array(K5,dtype=float)/16
    kernal6 = np.array(K6,dtype=float)/16

    #convolve the image
    dst = cv2.filter2D(diag1,-1,kernal6)

    #pad the image
    pad = np.pad(diag1,2,'constant')

    #center val
    val = abs(dst[3][3])


    print("original")
    print(diag1)
    print("convolved")
    print(dst)
    print("padded")
    print(pad)

    print("center val: %s" % val)



def create_histogram():
    image = np.array([2,4,5,6])

    plt.plot(image,color='black')#   show flag
    plt.show()



#create_histogram()
#test_getslice()
#test_extractEdgeDistribution("lena3.jpg","SHOW")
#test_extractColorDistribution("lena3.jpg","SHOW")
#test_getSegments("lena3.jpg","SHOW")
#test_getSegments("/home/masa/Projects/TrashClassification_capstone/Source/MixedB30_C30_T30Sample1.jpg","SHOW")


