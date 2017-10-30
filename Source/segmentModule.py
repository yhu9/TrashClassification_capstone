#############################################################################################################
#Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#Library imports
import numpy as np
import cv2
import random
import math
import Tkinter as tk
import pymeanshift as pms
from matplotlib import pyplot as plt
############################################################################################################
#Flag options for imread are self explanatory
#cv2.IMREAD_GRAYSCALE
#Cv2.IMREAD_COLOR
#cv2.IMREAD_UNCHANGED
#############################################################################################################
#Global Variables
allimages = {}                          #put all images in this dictionary here to show them later
#############################################################################################################

MIN_DENSITY = 500

def getSegments(imageIn, SHOW):
    original = cv2.imread(imageIn,cv2.IMREAD_COLOR)
    allimages["original"] = original
    ##############################################################################################################
    #Gray image
    #gray_img = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
    #allimages["gray"] = gray_img

    #HSV image
    #hsv_img = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
    #allimages["hsv"] = hsv_img

    #gaussian Blur
    #blur = cv2.GaussianBlur(gray_img,(5,5),0)
    #allimages["gaussianBlur"] = blur

    #hsv Blur
    #hsv_blur = cv2.GaussianBlur(hsv_img,(5,5),0)

    #Binarize Image with OTSU's Algorithm
    #ret, binImg = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #allimages["binImage"] = binImg

    #mean shift segmentation on bgr image
    #https://github.com/fjean/pymeanshift
    (segmented_image,labels_image,number_regions) = pms.segment(
            original,
            spatial_radius=6,
            range_radius=4.5,
            min_density=MIN_DENSITY)
    print(number_regions)
    unique_labels = np.unique(labels_image)
    blank = original - original
    for label in unique_labels:
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        blank[ labels_image == label] = [b,g,r]
    allimages["shift segmentation"] = blank


    #laplacian filtering
    #laplacian = cv2.Laplacian(gray_img,cv2.CV_32F)
    #allimages["laplacian"] = laplacian

    #blur laplacian
    #blur_lap = cv2.GaussianBlur(laplacian,(5,5),10)
    #allimages["blurred lap"] = blur_lap

    #sobel
    #sobelx = cv2.Sobel(blur_lap,cv2.CV_32F,1,0,ksize=5)
    #sobely = cv2.Sobel(blur_lap,cv2.CV_32F,0,1,ksize=5)
    #mag, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
    #allimages["sobel y"] = sobely
    #allimages["sobel x"] = sobelx
    #allimages["sobel maginitude"] = angle

    #edge detection using canny
    #max_threshold = ret
    #min_threshold = max_threshold * 0.5
    #edges = cv2.Canny(blur,min_threshold,max_threshold)
    #allimages["canny"] = edges

    #findContours(src, Hierarchy, Optimization)
    #RETR_LIST makes flat hierarchy of contours found
    #drawContours(outputImg, contours, hierarchy, line width)
    #-1 draws all contours found
    #contourImg, contours, hierarchy = cv2.findContours(binImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(contourImg,contours, -1, 255,3)
    #contourImg = cv2.bitwise_not(contourImg)
    #allimages["find Contours"] = contourImg

    #Find and Mark Connected Components separately. All spots found are marked > 0
    #ret, markers = cv2.connectedComponents(contourImg)

    ################################################################################
    #Marked spots are not filled completely with water. Unknown spots marked as 0 and will be filled with water first
    #Watershed algorithm using the markers. Unfilled spots after the watershed algorithm is marked with a -1
    #markers = cv2.watershed(original,markers)

    #Apply markers on black image
    #binImg = original.copy()
    #binImg = binImg - binImg
    #for x in range(np.max(markers) + 1):
    #    b = random.randint(0,255)
    #    g = random.randint(0,255)
    #    r = random.randint(0,255)
    #    binImg[markers == x] = [b,g,r]
    #allimages["connected components"] = binImg

    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    if SHOW:

        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        def quit():
            root.destroy()
        quit()
        if len(allimages) < 5:
            width = int(width / 2)
            height = int(height / 2)
            x,y = 0,0
            imgCount = 1
            for key,val in allimages.items():
                if imgCount > 2:
                    row = 1
                else:
                    row = 0
                if imgCount % 2 == 1:
                    col = 0
                else:
                    col = 1
                cv2.namedWindow(key,cv2.WINDOW_NORMAL)
                cv2.imshow(key,val)
                cv2.resizeWindow(key,width,height)
                cv2.moveWindow(key, width * col, height * row)
                imgCount += 1
        ########################################################
        #The else isn't ever used but I left it since more images may want to be added during a SHOW
        else:
            width = int(width / 3)
            height = int(height / 3)
            x,y = 0,0
            imgCount =0
            for key,val in allimages.items():
                row = int(imgCount % 3)
                col = int(math.floor(imgCount / 3))
                cv2.namedWindow(key,cv2.WINDOW_NORMAL)
                cv2.resizeWindow(key,width,height)
                cv2.moveWindow(key, width * col, height * row)
                cv2.imshow(key,val)
                imgCount += 1
        cv2.waitKey(0)
        #There is a bug that makes it so that you have to close windows like this on ubuntu 12.10 sometimes.
        #http://code.opencv.org/issues/2911
        cv2.destroyAllWindows()
        cv2.waitKey(-1)
        cv2.imshow('',original)

    return original, labels_image

###############################################################################################################################
###############################################################################################################################
#Documentation
########################################################################
#BilateralFilter
########################################################################
#http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
########################################################################
#Prameters:
#    src - src image
#    dst - Destination image of the same size and type as src .
#    d - Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
#    sigmaColor - Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
#    sigmaSpace - Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .
#
# bilateralFilter(src, d, sigmaColor, sigmaSpace)


########################################################################

########################################################################
#Canny image
#http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
########################################################################
#Parameters:
#    image - single-channel 8-bit input image.
#    edges - output edge map; it has the same size and type as image .
#    threshold1 - first threshold for the hysteresis procedure.
#    threshold2 - second threshold for the hysteresis procedure.
#    apertureSize - aperture size for the Sobel() operator.
#    L2gradient - a flag, indicating whether a more accurate L_2 norm =\sqrt{(dI/dx)^2 + (dI/dy)^2} should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L_1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).
