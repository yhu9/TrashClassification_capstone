#!/usr/bin/python
#############################################################################################################
#Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#Library imports
import sys
import segmentModule
import extractionModule
import numpy as np
import cv2

#############################################################################################################
#Check system argument length
if len(sys.argv) == 4:
    imageFileIn = str(sys.argv[1])
    fnameout = str(sys.argv[2])
    showFlag = (str(sys.argv[3]) == "show")

    if len(imageFileIn) == 0 or len(fnameout) == 0:
        print("image file name or file name out is empty")
        sys.exit()

    image, markers = segmentModule.getSegments(imageFileIn,showFlag)
    features = extractionModule.extractFeatures(image,markers,showFlag)
    extractionModule.writeFeatures(features,fnameout)
    
else:
    print "wrong number of files as arguments"
    sys.exit()

