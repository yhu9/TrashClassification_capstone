import sys
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

#Flag options for imread are
#cv2.IMREAD_GRAYSCALE
#Cv2.IMREAD_COLOR
#cv2.IMREAD_UNCHANGED
original =  cv2.imread('/home/masa/Documents/Projects/TrashClassification_capstone/Source/prettyflower.jpg',cv2.IMREAD_COLOR)
fout = 'outputImages/'



plt.figure(1)
plt.imshow(original)
plt.colorbar()
 
plt.figure(2)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([original],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])


    print("color is: %s" % col)
    print("--------------------------------------------------------------------------")
    for x in histr:
        string = str(x[0]) + ","
        sys.stdout.write(string)

    print("\n")

if sys.argv[1] == "show":
    plt.show()
    plt.close('all')

##############################################################################################################
#Gray image
gray_img = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
#Blur image
blurred = cv2.blur(gray_img,(20,20))
#Binarize Image with OTSU's Algorithm
ret, thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#Find and Mark Connected Components separately. All spots found are marked > 0
ret, markers = cv2.connectedComponents(thresh)
#Watershed algorithm using the markers. Unfilled spots are marked as -1
markers = cv2.watershed(original,markers)
#Apply the markers on original image
img = original.copy()
img[markers == -1] = [0,0,255]
#Apply markers on black image
blackimg = img.copy()
blackimg = blackimg - blackimg
blackimg[markers == -1] = [0,0,255]
#Color each connected component a random color
for x in range(np.max(markers) + 1):
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
    blackimg[markers == x] = [b,g,r]

if sys.argv[1] == "show":
    cv2.namedWindow('orignal',cv2.WINDOW_NORMAL)
    cv2.imshow('original',original)
    cv2.namedWindow('grey scaled',cv2.WINDOW_NORMAL)
    cv2.imshow('grey scaled',gray_img)
    cv2.namedWindow('blurred',cv2.WINDOW_NORMAL)
    cv2.imshow('blurred',blurred)
    cv2.namedWindow('threshed',cv2.WINDOW_NORMAL)
    cv2.imshow("threshed",thresh)
    cv2.namedWindow('results', cv2.WINDOW_NORMAL)
    cv2.imshow('results',img)
    cv2.namedWindow('colored blobs',cv2.WINDOW_NORMAL)
    cv2.imshow('colored blobs',blackimg)


    cv2.imwrite(fout + 'original.png',orignal)
    cv2.imwrite(fout + 'grayedImg.png',gray_img)
    cv2.imwrite(fout + 'blurredImg.png',blurred)
    cv2.imwrite(fout + 'binaryImg.png',thresh)
    cv2.imwrite(fout + 'resuts.png',img)
    cv2.imwrite(fout + 'blobs.png',blackimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

