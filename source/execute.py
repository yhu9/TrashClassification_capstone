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
if len(sys.argv) >= 4:
    imageFileIn = str(sys.argv[1])
    fnameout = str(sys.argv[2])
    showFlag = (str(sys.argv[3]) == "show")
    mode = str(sys.argv[4])

    if len(imageFileIn) == 0 or len(fnameout) == 0:
        print("image file name or file name out is empty")
        sys.exit()

    output = segmentModule.normalizeImage(imageFileIn)
    image, markers = segmentModule.getSegments(output,showFlag)
    features = extractionModule.extractFeatures(image,markers,mode,showFlag)
    extractionModule.writeFeatures(features,fnameout)

else:
    print "wrong number of files as arguments expecting 3:"
    print "argv1 = imageFileIn"
    print "argv2 = fnameout"
    print "argv3 = showFlag (must be show/noshow)"
    print "argv4 = mode (color,edge,both)"
    sys.exit()

