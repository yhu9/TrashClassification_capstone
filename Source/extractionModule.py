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
K3 = [[2,2,-1],[2,-1,-1],[-1,-1,-1]]      #diagnol1
K4 = [[-1,2,2],[-1,-1,2],[-1,-1,-1]]      #diagnol2
KERNAL1 = np.array(K1,dtype=float)/8
KERNAL2 = np.array(K2,dtype=float)/8
KERNAL3 = np.array(K3,dtype=float)/8
KERNAL4 = np.array(K4,dtype=float)/8
#KERNAL5 = np.array(K5,dtype=float)/16
#KERNAL6 = np.array(K6,dtype=float)/16

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
def extractFeatures(imageIn,markers,mode,SHOW):
    features = []
    uniqueMarkers = np.unique(markers)
    time = -1
    if SHOW:
        time = int(raw_input("How quickly do you want to see histogram creation? Type time in ms: "))

    #skip 0 because thats the marker for uknown areas separating the regions
    print "-----------Unique Markers------------"
    print "-------------------------------------"
    print ("number of unique markers: %i" % len(uniqueMarkers))

    if(mode == 'color'):
        for x in uniqueMarkers[1:]:
            feature = extractColorDistribution(imageIn,x,markers,SHOW,time)
            features.append(feature)
    elif(mode == 'edge'):
        for x in uniqueMarkers[1:]:
            feature = extractEdgeDistribution(imageIn,x,markers,SHOW,time)
            features.append([feature])
    elif(mode == 'both'):
        for x in uniqueMarkers[1:]:
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
def extractColorDistribution(imageIn, uq_mark, markers, SHOW, time):
    plt.ion()       #sets plots to interactive mode

    #hsv = cv2.cvtColor(imageIn,cv2.COLOR_BGR2HSV)
    # make a copy of the hsv image
    region = imageIn.copy()

    #my hack for taking the subset of the image involves losing the color black on a HSV color scale
    #   -   all spots besides those marked uq_mark is made black
    region[markers != uq_mark] = [0,0,0]

    #############################################
    if SHOW:                                    #
        blank = region.copy()                      #   show flag
        blank = blank - blank                   #
        blank[markers == uq_mark] = [0,0,255]         #
        cv2.imshow('Processing Images',blank)   #
    #############################################
    #extract histogram distribution of each HSV value
    #output of calcHist is a np.array of length = ColorMax
    # H max = 170
    # S max = 255
    # V max = 255
    f = []
    colorRange = 0
    color = ('blue','green','red')
    for i,col in enumerate(color):
        if i == 0:
            colorRange = 255
        else:
            colorRange = 255

        histr = cv2.calcHist([region],[i],None,[colorRange],[1,colorRange])
        maxVal = np.max(histr)
        if maxVal != 0:
            histr = histr / maxVal
        else:
            histr = histr
            #print "0 val found"
        f.append(histr.flatten())
        ##################################
        if SHOW:                         #
            plt.plot(histr,color=col)#   show flag
            plt.draw()
            plt.pause(0.00001)
        ##################################
    #############################
    if SHOW:                    #
        cv2.waitKey(time)   #   show flag
        plt.clf()           #
    #############################

    plt.ioff()       #turns off interactive mode
    return f

#extract the edge distribution from the image segment
def extractEdgeDistribution(imageIn, uq_mark, markers, SHOW, time):

    gray = cv2.cvtColor(imageIn,cv2.COLOR_BGR2GRAY)

    # make a copy of the hsv image
    region = gray.copy()

    #all spots besides those marked uq_mark is made black and uq_mark is made white
    region[markers != uq_mark] = [0]
    region[markers == uq_mark] = [255]

    #detect edges perhaps use thresholding here for automatic choice of threshold
    min_threshold = 100
    max_threshold = 200
    edges = cv2.Canny(region,min_threshold,max_threshold)

    #pad the image
    padded_region = np.pad(region,10,'constant')
    padded_edges = np.pad(edges,10,'constant')

    #get edge locations ((x1,x2,x3,...),(y1,y2,y3,...))
    edge_locations = np.where(padded_edges == 255)

    if(SHOW):
        cv2.imshow('region of interest',region)
        cv2.imshow('detected edges',edges)

    #initialize kernal counting histogram
    histogram = [0,0,0,0]
    ksize = 3
    for x,y in zip(edge_locations[0],edge_locations[1]):

        #get window slice with edge_location as center
        low = (int)(ksize/2)
        high = ksize - low
        window_slice = padded_region[x-low:x+high,y-low:y+high]

        #use the kernals
        #convolution applied to all pixels in the slice and I can't stop it.
        filters = []
        filter1 = cv2.filter2D(window_slice,-1,KERNAL1)
        filter2 = cv2.filter2D(window_slice,-1,KERNAL2)
        filter3 = cv2.filter2D(window_slice,-1,KERNAL3)
        filter4 = cv2.filter2D(window_slice,-1,KERNAL4)
        filters.append(filter1)
        filters.append(filter2)
        filters.append(filter3)
        filters.append(filter4)

        #get center value of the filtered slice
        cx = (int)(ksize/2)
        cy = cx
        vals = []
        for f in filters:
            val = abs(f[cx][cy])
            vals.append(val)

        #get the index of the max val
        index = vals.index(max(vals))

        #increment the histogram
        histogram[index] += 1


    #normalize the values in the histogram with the max so values fall between 0-1
    feature = np.array(histogram,np.float32)/max(histogram)

    if(SHOW):                         #
        plt.plot(feature,color='red')#   show flag
        plt.draw()
        cv2.waitKey(time)   #   show flag
        plt.show()


    return feature


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

