
#Imports
import segmentModule
import numpy as np
import sys
import cv2

#Constants
INGROUP=np.array([255,255,255],np.uint8)
OUTGROUP=np.array([0,0,0],np.uint8)
UNKNOWN=np.array([125,125,125],np.uint8)
GROUP1=np.array([0,0,255],np.uint8)
GROUP2=np.array([0,255,0],np.uint8)
GROUP3=np.array([255,0,0],np.uint8)
GROUP4=np.array([255,255,255],np.uint8)

GROUPS = [GROUP1,GROUP2,GROUP3,GROUP4,UNKNOWN]
SINGLE = [INGROUP,OUTGROUP,UNKNOWN]

#Local functions


#findMax
# - Finds the index of the maximum value in the list
def findMax(args):
    index = 0
    tmp = 0
    if type(args) != type([]):
        return -1

    for i,a in enumerate(args):
        if a > tmp:
            index = i
            tmp = a

    return index


#main program
#

#When using a single classification
if len(sys.argv) == 3:
    imageFileIn = sys.argv[1]
    classificationsIn = sys.argv[2]

    #initialize image, markers using segmentModule
    #initialize classifications using classificationsIn
    #segmentModule.getSegments always produces the same result so this works. Since classification for each segment is known using same function in execute.py.
    image, markers = segmentModule.getSegments(imageFileIn, False)
    uniquemarkers = np.unique(markers)
    classifications = []
    with open(classificationsIn,'r') as cin:
        lines = cin.read().splitlines()
        for l in lines:
            classifications.append(float(l))

    if len(classifications) == len(uniquemarkers[1:]):

        blank = image.copy()
        blank = blank - blank
        blank[markers == -1] = UNKNOWN
        for c,um in zip(classifications,uniquemarkers[1:]):
            if  c > 0:
                blank[markers == um] = INGROUP
            elif c <= 0:
                blank[markers == um] = OUTGROUP
        
        
        total = 0
        pixcounts = []
        for group in SINGLE:
            tmp = cv2.inRange(blank,group,group)
            num = cv2.countNonZero(tmp)
            pixcounts.append(num)
            total += num

        percent1 = float(pixcounts[0]) / float(total) * 100
        percent2 = float(pixcounts[1]) / float(total) * 100
        percent3 = float(pixcounts[2]) / float(total) * 100

        cv2.namedWindow(imageFileIn,cv2.WINDOW_NORMAL)
        cv2.namedWindow("original", cv2.WINDOW_NORMAL)
        print ""
        print("ingroup white:  %.5f%%" % percent1)
        print("outgroup black: %.5f%%" % percent2)
        print("unkown grey:    %.5f%%" % percent3)
        print ""

        cv2.imshow("original",image)
        cv2.imshow(imageFileIn,blank)
        cv2.waitKey(0)

    else:
        print "Are you sure the classification file is for that image?"

#When using multiple classifications. No more than 4
elif len(sys.argv) > 3 and len(sys.argv) <= 6:
    #get command line arguments
    imageFileIn = sys.argv[1]
    classification_names = []
    for fname in sys.argv[2:]:
        classification_names.append(fname)
    
    #recreate markers
    image, markers = segmentModule.getSegments(imageFileIn, False)
    uniquemarkers = np.unique(markers)
    
    #get classfications for segments from command line arguments
    classifications = []
    for fname in classification_names:
        classifications.append([])
        with open(fname,'r') as fin:
            lines = fin.read().splitlines()
            for l in lines:
                classifications[-1].append(float(l))

    #If classifications are for the same image, then start stitching according to best classifier.
    if len(classifications[0]) == len(classifications[1]) and len(classifications[2]) == len(classifications[3]) and len(classifications[1]) == len(classifications[2]):
        
        #make blank image
        blank = image.copy()
        blank = blank - blank
        blank[markers == -1] = UNKNOWN

        #Color using max of the classifications. The max according to the svm is the one farthest from the Support vector line separating the different 2 group classification. The most positive is taken
        for c1,c2,c3,c4,um in zip(classifications[0],classifications[1],classifications[2],classifications[3],uniquemarkers[1:]):
            tmp = [c1,c2,c3,c4]
            index = findMax(tmp)
            #color according to the group index
            blank[markers == um] = GROUPS[index]

        pixcounts = []
        total = 0
        for g in GROUPS:
            tmp = cv2.inRange(blank,g,g)
            num = cv2.countNonZero(tmp)
            pixcounts.append(num)
            total += num
        
        percent1 = float(pixcounts[0]) / float(total) * 100
        percent2 = float(pixcounts[1]) / float(total) * 100
        percent3 = float(pixcounts[2]) / float(total) * 100
        percent4 = float(pixcounts[3]) / float(total) * 100
        percent5 = float(pixcounts[4]) / float(total) * 100
        
        cv2.namedWindow(imageFileIn,cv2.WINDOW_NORMAL)
        cv2.namedWindow("original", cv2.WINDOW_NORMAL)
        print ""
        print (" construction waste (red): %.5f%%" % percent1)
        print ("      tree matter (green): %.5f%%" % percent2)
        print ("     general goods (blue): %.5f%%" % percent3)
        print ("       trash bags (white): %.5f%%" % percent4)
        print ("segment separators (grey): %.5f%%" % percent5)
        print ""
        cv2.imshow("original",image)
        cv2.imshow(imageFileIn,blank)
        cv2.waitKey(0)

    else:
        print "classifications are not the same length"

else:
    print "wrong number of arguments passed. Expecting 2 or 5:"
    print "arg1 = imageFileIn"
    print "arg2 = classification1"
    print "arg3 = classification2"
    print "arg4 = classification3"
    print "arg5 = classification4"
