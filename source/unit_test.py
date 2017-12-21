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


def create_histogram():
    image = np.array([2,4,5,6])

    plt.plot(image,color='black')#   show flag
    plt.show()

####################################################################################################
def test_playground():
    print "hello world"

#test_playground()
#test_extractEdgeDistribution("categories/test_images/TestImages/Square.png","SHOW")
#test_extractColorDistribution("categories/test_images/TestImages/Circle.png","SHOW")
#test_extractEdgeDistribution("lena3.jpg","SHOW")
#test_extractColorDistribution("lena3.jpg","SHOW")
#test_getSegments("lena3.jpg","SHOW")
#test_getSegments("/home/masa/Projects/TrashClassification_capstone/Source/MixedB30_C30_T30Sample1.jpg","SHOW")

test_getSegments("/home/masa/Projects/TrashClassification_capstone/Source/CervicalCells/image1.png","SHOW")

#test_extractColorDistribution("/home/masa/Projects/TrashClassification_capstone/Source/categories/treematter/ingroup/treematter1.jpg","SHOW")

#test_extractEdgeDistribution("/home/masa/Projects/TrashClassification_capstone/Source/categories/test_images/Rect.png","SHOW")
#test_extractEdgeDistribution("/home/masa/Projects/TrashClassification_capstone/Source/categories/test_images/Rect2.png","SHOW")
#test_extractEdgeDistribution("/home/masa/Projects/TrashClassification_capstone/Source/categories/test_images/Square.png","SHOW")
#test_extractEdgeDistribution("/home/masa/Projects/TrashClassification_capstone/Source/categories/test_images/Circle.png","SHOW")
#test_extractEdgeDistribution("/home/masa/Projects/TrashClassification_capstone/Source/categories/test_images/Diamond.png","SHOW")
#test_extractEdgeDistribution("/home/masa/Projects/TrashClassification_capstone/Source/categories/test_images/Star.png","SHOW")

