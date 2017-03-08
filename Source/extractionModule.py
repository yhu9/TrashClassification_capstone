#############################################################################################################
#Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#Library imports
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
#############################################################################################################
#                               Description of Module
#This module takes in the inputs:
#   Mat         image
#   np.array    markers
#   bool        SHOW
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

#According to http://stackoverflow.com/questions/17063042/why-do-we-convert-from-rgb-to-hsv/17063317
#HSV is better for object recognition compared to BGR 
def extractFeatures(imageIn, markers, SHOW):
	features = []
        plt.ion()       #sets plots to interactive mode
	uniqueMarkers = np.unique(markers)
	hsv = cv2.cvtColor(imageIn,cv2.COLOR_BGR2HSV)
	print "-1 is unknown regions"
	print "---------------------------------------------------"
	print "-----------Unique Markers------------"
	print uniqueMarkers
        if SHOW:
            time = int(raw_input("How quickly do you want to see histogram creation? Type time in ms: "))

	#skip -1 because thats the marker for unkown areas separating the regions
        #my hack for taking the subset of the image involves losing the color black on a HSV color scale
        #   -   region is the same size as the original image except that all spots besides those marked x
        #       is made black
        #   -   blank is a blank canvas the same size as the orginal image except that the region marked x
        #       is made red
	for x in uniqueMarkers[1:]:
		region = hsv.copy()
		region[markers != x] = [0,0,0]
                #############################################
                if SHOW:                                    #
		    blank = hsv.copy()                      #   show flag
		    blank = blank - blank                   #
		    blank[markers == x] = [0,0,255]         #
                    cv2.imshow('Processing Images',blank)   #
                #############################################
                #extract histogram distribution of each HSV value
                #output of calcHist is a np.array of length = ColorMax
                # H max = 170
                # S max = 255
                # V max = 255
                f = []
                colorRange = 0
		for i in range(3):
                    if i == 0:
                        colorRange = 179
                    else:
                        colorRange = 255

		    histr = cv2.calcHist([region],[i],None,[colorRange],[1,colorRange])
                    maxVal = np.max(histr)
                    histr = histr / maxVal
                    f.append(histr.flatten())
                    ##################################
                    if SHOW:                         #
                        plt.plot(histr,color='black')#   show flag
                        plt.draw()                   #
                    ##################################
                #############################
                if SHOW:                    #
		        cv2.waitKey(time)   #   show flag
                        plt.clf()           #
                #############################
                features.append(f)

        plt.ion()       #turns off interactive mode
	return features


#Writes the features out to a file called extractionOut/features.txt
def writeFeatures(features, fnameout):
    if (len(features) == 0) or (type(features) != type([])):
        print ("features: %s" % type(features))
        print ("expected: %s" % type([]))
        print ("length features: %i" % len(features))
        print "error with the input to the extractionModule.writeFeatures()"
        print "expecting a populated list of np.arrays"
        return False
    else:
        with open(fnameout, 'w') as fout:
            for f in features:
                for histogram in f:
                    for val in histogram:
                        fout.write(",")
                        fout.write(str(val))
                    #fout.write(",".join(map(str,histogram)))
                fout.write('\n')

        return True




