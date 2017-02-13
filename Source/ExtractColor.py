import numpy as np
import cv2
from matplotlib import pyplot as plt

#Flag options for imread are
#cv2.IMREAD_GRAYSCALE
#Cv2.IMREAD_COLOR
#cv2.IMREAD_UNCHANGED
img =  cv2.imread('/home/masa/Documents/Projects/TrashClassification_capstone/Source/prettyflower.jpg',cv2.IMREAD_COLOR)

plt.figure(1)
plt.imshow(img)
plt.colorbar()
 
plt.figure(2)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

plt.show()



