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

MIN_DENSITY = 50
SPATIAL_RADIUS = 8
RANGE_RADIUS = 6

WIDTH = 128 * 4
HEIGHT = 128 * 4

def normalizeImage(imageIn):
    original = cv2.imread(imageIn,cv2.IMREAD_COLOR)

    resized = cv2.resize(original,(WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)

    return resized

#saves the segments of the original image as png files given the labels
def saveSegments(original,labels,SHOW,out_dir,category):

    unique_labels = np.unique(labels)

    #get the sizes of the discovered segments
    size_array = []
    size_dict = {}
    for x in unique_labels:
        count = np.count_nonzero(labels == x)
        size_array.append(count)
        size_dict[x] = count

    #get information about the segments
    mean = np.mean(size_array)
    total = np.sum(size_array)
    count = len(unique_labels)

    #remove markers given condition ange get unique markers again
    for k in size_dict.keys():
        if(size_dict[k] < mean):
            labels[labels == k] = 0
    unique_labels = np.unique(labels)
    reduced_count = len(unique_labels)

    count = 0
    for l in unique_labels[1:]:
        segment = original.copy()
        segment[labels != l] = [0,0,0]

        blank = original.copy()
        blank = blank - blank
        blank[labels == l] = [255,255,255]

        grey = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
        x,y,w,h = cv2.boundingRect(grey)
        cropped = segment[y:y+h,x:x+w]
        cropped = np.uint8(cropped)
        resized = cv2.resize(cropped,(256, 256), interpolation = cv2.INTER_CUBIC)
        img = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)

        f_out =  out_dir + category + str(count) + '.png'

        fig = plt.figure()
        plt.imshow(img)
        fig.savefig(f_out)
        if(SHOW):
            plt.show()
        plt.close(fig)

        count = count + 1

    print("reduced count: " + count)

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
def getSegments(original, SHOW):
    allimages["original"] = original
    ##############################################################################################################

    #gaussian Blur
    #blur = cv2.GaussianBlur(gray_img,(5,5),0)
    #allimages["gaussianBlur"] = blur

    #mean shift segmentation on bgr image
    #https://github.com/fjean/pymeanshift
    #http://ieeexplore.ieee.org/document/1000236/
    (segmented_image,labels_image,number_regions) = pms.segment(
            original,
            spatial_radius=SPATIAL_RADIUS,
            range_radius=RANGE_RADIUS,
            min_density=MIN_DENSITY)
    print("Number of Regions Found: %s" % number_regions)
    unique_labels = np.unique(labels_image)
    blank = original - original
    for label in unique_labels:
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        blank[ labels_image == label] = [b,g,r]

    allimages["shift segmentation"] = blank

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

    return original, labels_image

