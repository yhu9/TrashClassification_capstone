import segmentModule
import numpy as np
import sys
import cv2

#Constants
INGROUP=[0,0,0]
OUTGROUP=[255,255,255]
UNKNOWN=[125,125,125]

#main program
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
        cv2.namedWindow(imageFileIn,cv2.WINDOW_NORMAL)

        blank = image.copy()
        blank = blank - blank
        blank[markers == -1] = UNKNOWN
        for c,um in zip(classifications,uniquemarkers[1:]):
            if  c > 0:
                blank[markers == um] = INGROUP
            elif c <= 0:
                blank[markers == um] = OUTGROUP
        
        num_ingroup = (blank == INGROUP).sum()
        num_outgroup = (blank == OUTGROUP).sum()
        num_unknown = (blank == UNKNOWN).sum()
        total = num_ingroup + num_outgroup + num_unknown
        
        percentIn = 0
        if total != 0:
            percentIn = float(num_ingroup + num_unknown) / float(total) * 100
        else:
            percentIn = -1.0

        print ""
        print ("percentage of the image in the category: %.3f%%" % percentIn)
        print ""

        cv2.imshow(imageFileIn,blank)
        cv2.waitKey(0)

    else:
        print "Are you sure the classification file is for that image?"

else:
    print "wrong number of arguments passed. Expecting 2:"
    print "arg1 = imageFileIn"
    print "arg2 = classificationIn"
