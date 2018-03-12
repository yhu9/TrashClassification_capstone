#############################################################################################################
#Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#Library imports
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

#############################################################################################################
#                               Edge Distribution Masks
#create the kernals
#not sure what the 5x5 version of the diagnol is.
#http://www.cim.mcgill.ca/~image529/TA529/Image529_99/assignments/edge_detection/references/sobel.htm
K1 = [[1,2,1],[0,0,0],[-1,-2,-1]]         #horizontal
K2 = [[1,0,-1],[2,0,-2],[1,0,-1]]         #vertical
K3 = [[2,2,-1],[2,0,-1],[-1,-1,-1]]      #diagnol1
K4 = [[-1,2,2],[-1,0,2],[-1,-1,-1]]      #diagnol2
K5 = [[-1,-1,-1],[-1,0,2],[-1,2,2]]      #diagnol3
K6 = [[-1,-1,-1],[2,0,-1],[2,2,-1]]      #diagnol4
KERNAL1 = np.array(K1,dtype=float)/8
KERNAL2 = np.array(K2,dtype=float)/8
KERNAL3 = np.array(K3,dtype=float)/8
KERNAL4 = np.array(K4,dtype=float)/8
KERNAL5 = np.array(K5,dtype=float)/8
KERNAL6 = np.array(K6,dtype=float)/8

#############################################################################################################
#                               Description of Module
#
#The output of the exported extract features function is:
#   array([np.array,np.array,np.array], ...]    features
#
#brief description:
#
#this module takes a source image with marked regions and extracts HSV color histograms
#as features for each region defined in markers. The HSV color value [0,0,0] gets dropped
#due to constraints on the opencv calcHist() function which must take a rectangular image
#as the input parameter. Since, marked regions are not rectangular, a copy of the original image
#is used with a particular marking to make an image that is all black except for the specified
#region. This image is then used to extract the histogram distribution of that region. This process
#is repeated until all regions are stored in features.
#
#Setting the show flag allows the user to specify how slowly they would like to see the histogram
#distribution extraction for each region.
#############################################################################################################
#############################################################################################################
def extractAllSegments(imageIn,markers,mode,SHOW):
    features = []
    uniqueMarkers = np.unique(markers)
    time = 0

    #skip 0 because thats the marker for uknown areas separating the regions
    print "-----------Unique Markers------------"
    print "-------------------------------------"
    print ("number of unique markers: %i" % len(uniqueMarkers))

    if(mode == 'color'):
        for x in uniqueMarkers:
            feature = extractColorDistribution(imageIn,x,markers,SHOW,time)
            features.append([feature])
    elif(mode == 'edge'):
        for x in uniqueMarkers:
            feature = extractEdgeDistribution(imageIn,x,markers,SHOW,time)
            features.append([feature])
    elif(mode == 'both'):
        for x in uniqueMarkers:
            feature1 = extractColorDistribution(imageIn,x,markers,SHOW,time)
            feature2 = extractEdgeDistribution(imageIn,x,markers,SHOW,time)

            #create feature vector as concatenation of feature 1 and feature 2
            feature = np.append(feature1,feature2)
            features.append([feature])
    return features

#for extracting high quality features to train on using selected segments from the image
#1. segments must greater than the average size
def extractFeatures(imageIn,markers,mode,SHOW):
    features = []
    uniqueMarkers = np.unique(markers)

    #get the sizes of the discovered segments
    size_array = []
    size_dict = {}
    for x in uniqueMarkers:
        count = np.count_nonzero(markers == x)
        size_array.append(count)
        size_dict[x] = count

    #get information about the segments
    mean = np.mean(size_array)
    total = np.sum(size_array)
    count = len(uniqueMarkers)

    #remove markers given condition ange get unique markers again
    for k in size_dict.keys():
        if(size_dict[k] < mean):
            markers[markers == k] = 0
    uniqueMarkers = np.unique(markers)
    reduced_count = len(uniqueMarkers)

    #show the segmenting size selection process
    if SHOW:
        print("mean size: %s" % mean)
        print("segment counts: %s" % count)
        print("reduced counts: %s" % reduced_count)
        size_array.sort()
        size_hist = np.array(size_array)
        subset = size_hist[size_hist > mean]
        plt.figure(1)
        plt.subplot(211)
        plt.plot(size_hist,'r--')

        plt.subplot(212)
        plt.plot(subset,'r--')
        plt.pause(0.1)
        cv2.waitKey(0)

    time = 0

    #skip 0 because thats the marker for uknown areas separating the regions
    print "-----------Unique Markers------------"
    print "-------------------------------------"
    print ("number of unique markers: %i" % len(uniqueMarkers))

    if(mode == 'color'):
        for x in uniqueMarkers:
            feature = extractColorDistribution(imageIn,x,markers,SHOW,time)
            features.append([feature])
    elif(mode == 'edge'):
        for x in uniqueMarkers:
            feature = extractEdgeDistribution(imageIn,x,markers,SHOW,time)
            features.append([feature])
    elif(mode == 'both'):
        for x in uniqueMarkers:
            feature1 = extractColorDistribution(imageIn,x,markers,SHOW,time)
            feature2 = extractEdgeDistribution(imageIn,x,markers,SHOW,time)

            #create feature vector as concatenation of feature 1 and feature 2
            feature = np.append(feature1,feature2)
            features.append([feature])

    return features

#The function extractFeatures() takes in the inputs:
#   Mat         image
#   np.array    markers
#   bool        SHOW
#
#The output of the exported extract features function is a 1-d np array
#
#According to http://stackoverflow.com/questions/17063042/why-do-we-convert-from-rgb-to-hsv/17063317
#HSV is better for object recognition compared to BGR
# H max = 170
# S max = 255
# V max = 255
def extractColorDistribution(imageIn, uq_mark, markers, SHOW, time):

    plt.ion() #sets plots to interactive mode
    #https://en.wikipedia.org/wiki/Color_histogram
    #possible colors = bin^3
    bins = 8
    colors = np.zeros((bins,bins,bins))

    #get the pixels of interest
    region_pixels = imageIn[markers == uq_mark]

    #create the histogram of n^3 colors
    for pixel in region_pixels:
        #color = ('blue','green','red')
        b = pixel[0]
        g = pixel[1]
        r = pixel[2]

        bbin = int(float(b) / float(256/bins))
        gbin = int(float(g) / float(256/bins))
        rbin = int(float(r) / float(256/bins))

        colors[bbin][gbin][rbin] += 1

    #flatten the 3-d feature fector into 1-d
    hist = colors.flatten()
    hist = hist / np.amax(hist)

    #show the results
    if SHOW:
        region = imageIn.copy()
        region[markers != uq_mark] = [0,0,0]
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',region)   #
        plt.plot(hist)
        plt.draw()
        plt.pause(0.1)
        cv2.waitKey(time)   #   show flag
        plt.clf()

    plt.ioff()

    return hist

#extract the edge distribution from the image segment
def extractEdgeDistribution(imageIn, uq_mark, markers, SHOW, time):
    #necessary for seeing the plots in sequence with one click of a key
    plt.ion() #sets plots to interactive mode

    # make a copy of the image
    region = imageIn.copy()
    region[markers != uq_mark] = [0,0,0]

    #get just the region and remove the rest
    blank = region.copy()                      #   show flag
    blank = blank - blank                   #
    blank[markers == uq_mark] = [255,255,255]         #

    #get the bounding rectangle of the image crop the region with a green border
    grey = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
    x,y,w,h = cv2.boundingRect(grey)
    cropped = region[y:y+h,x:x+w]
    cropped = np.uint8(cropped)

    new_w = ((int(w) / int(16)) + 1 ) * 16
    new_h = ((int(h) / int(16)) + 1 ) * 16

    #resize the image to 64 x 128
    resized = cv2.resize(cropped,(new_w, new_h), interpolation = cv2.INTER_CUBIC)
    height,width,channel = resized.shape

    #HOG DESCRIPTOR INITILIZATION
    #https://stackoverflow.com/questions/28390614/opencv-hogdescripter-python
    #https://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html
    #https://www.learnopencv.com/histogram-of-oriented-gradients/
    winSize = (width,height)                               #
    blockSize = (16,16)                             #only 16x16 block size supported for normalization
    blockStride = (8,8)                             #only 8x8 block stride supported
    cellSize = (8,8)                                #individual cell size should be 1/4 of the block size
    nbins = 9                                       #only 9 supported over 0 - 180 degrees
    derivAperture = 1                               #
    winSigma = 4.                                   #
    histogramNormType = 0                           #
    L2HysThreshold = 2.0000000000000001e-01         #L2 normalization exponent ex: sqrt(x^L2 + y^L2 + z^L2)
    gammaCorrection = 0                             #
    nlevels = 64                                    #
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                    histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

    hist = hog.compute(resized)

    #create the feature vector
    feature = []
    for i in range(nbins * 4):
        feature.append(0)
    for i in range(len(hist)):
        feature[i % (nbins * 4)] += hist[i]
    feature_hist = np.array(feature)
    feature_hist = feature / np.amax(feature)

    #show the results of the HOG distribution for the section
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',region)   #
        plt.plot(feature_hist)
        plt.draw()
        plt.pause(0.1)
        cv2.waitKey(time)   #   show flag
        plt.clf()

    plt.ioff()

    return feature_hist

#Writes the features out to a file called extractionOut/features.txt
def writeFeatures(features, fnameout):
    if (len(features) == 0) or (type(features) != type([])):
        print ("features: %s" % type(features))
        print ("expected: %s" % type([]))
        print ("length features: %i" % len(features))
        print "error with the input to the extractionModule.writeFeatures()"
        return False
    else:
        with open(fnameout, 'w') as fout:
            for f in features:
                for histogram in f:
                    for val in histogram:
                        fout.write(str(val))
                        fout.write(",")
                fout.write('\n')

        return True

