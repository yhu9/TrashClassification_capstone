import cv2
import numpy as np
import random
import os
import re
import math
import constants

#We try to read all the files all at once. Maybe I should fix that...
#inputs are 1d extracted feature vectors
#labels are one hot
#returns inputs and labels
def read_training_txt(directory):
    args = os.listdir(directory)
    inputs = []
    labels = []

    if(len(args) > 0):
        for f in args:
            full_dir = directory + f;
            img = cv2.imread(full_dir,cv2.IMREAD_COLOR)
            inputs.append(img)

            group = re.findall("treematter|construction|cardboard|plywood",full_dir)
            with open(f, 'r') as fin:
                lines = fin.read().splitlines()
                for l in lines:
                    row = []
                    if(group[0] == constants.CAT1):
                        labels.append(constants.CAT1_ONEHOT)
                    elif(group[0] == constants.CAT2):
                        labels.append(constants.CAT2_ONEHOT)
                    elif(group[0] == constants.CAT3):
                        labels.append(constants.CAT3_ONEHOT)
                    elif(group[0] == constants.CAT4):
                        labels.append(constants.CAT4_ONEHOT)
                    else:
                        print(group[0])

                    tokens = re.findall("\d+\.\d+",l)
                    for i,t in enumerate(tokens):
                        row.append(float(t))
                    inputs.append(row)
    else:
        print('error no file found')
        sys.exit()

    #shuffle the contents
    c = zip(inputs,labels)
    random.shuffle(c)
    inputs,labels = zip(*c)

    #use the last 1/5 of the inputs and labels for testing accuracy
    n = len(inputs)
    split = (int)(0.8 * float(n))
    train_set = inputs[:split]
    train_labels = labels[:split]
    test_set = inputs[split + 1:]
    test_labels = labels[split + 1:]

    #return the created content
    return np.array(train_set),np.array(train_labels),np.array(test_set),np.array(test_labels)

#reads in training images
#all training images must be passed when calling nn.py
def read_training_img(directory):
    inputs = []
    labels = []
    args = os.listdir(directory)

    if(len(args) > 0):
        for f in args:
            full_dir = directory + f;
            img = cv2.imread(full_dir,cv2.IMREAD_COLOR)
            resize = cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC)
            inputs.append(resize)

            group = re.findall("treematter|construction|cardboard|plywood",f)

            if(group[0] == constants.CAT1):
                labels.append(constants.CAT1_ONEHOT)
            elif(group[0] == constants.CAT2):
                labels.append(constants.CAT2_ONEHOT)
            elif(group[0] == constants.CAT3):
                labels.append(constants.CAT3_ONEHOT)
            elif(group[0] == constants.CAT4):
                labels.append(constants.CAT4_ONEHOT)
            else:
                print(group[0])

    else:
        print('error no file found')
        sys.exit()

    #shuffle the contents
    c = zip(inputs,labels)
    random.shuffle(c)
    inputs,labels = zip(*c)

    #use the last 1/5 of the inputs and labels for testing accuracy
    n = len(inputs)
    split = (int)(0.8 * float(n))
    train_set = inputs[:split]
    train_labels = labels[:split]
    test_set = inputs[split + 1:]
    test_labels = labels[split + 1:]

    #return the created content
    return np.array(train_set),np.array(train_labels),np.array(test_set),np.array(test_labels)


#reads in training image for cnn using pixel data as the training set
#28 x 28 surrounding area of each pixel used for training
#3x3 conv, 7x7 conv
#all training images must be passed when calling nn.py
def cnn_readOneImg(image_dir):

    img = cv2.imread(image_dir,cv2.IMREAD_COLOR)

    return(img)

def shrinkImg(img):

    resized = cv2.resize(img,(256,256),interpolation = cv2.INTER_CUBIC)

    return(resized)

#gets n patches from an image with its respective label
def getImgBatch(img,n):
    inputs = []
    labels = []

    rows = len(img)
    cols = len(img[0])
    depth = len(img[0][0])

    for i in range(n):
        a = random.randint(28,rows - 29)
        b = random.randint(28,cols - 29)

        #28x28 box on random point in the image
        box = img[a-14:a+14,b-14:b+14]

        if(a < 500 and b < 500):#topleft
            labels.append(constants.CAT1_ONEHOT)
        elif(a < 500 and b >= 500):#topright
            labels.append(constants.CAT2_ONEHOT)
        elif(a >= 500 and b < 500):#botleft
            labels.append(constants.CAT3_ONEHOT)
        elif(a >= 500 and b >= 500):#botright
            labels.append(constants.CAT4_ONEHOT)

        inputs.append(box)

    return inputs,labels

def read_unknown():
    inputs = []
    for f in sys.argv[1:]:
        if len(f) == 0:
            print("bad files passed")
            print("expected: ")
            print("file_directory_in received: " + f)
            sys.exit()

        with open(f, 'r') as fin:
            lines = fin.read().splitlines()
            for l in lines:
                tokens = re.findall("\d+\.\d+",l)
                for i,t in enumerate(tokens):
                    output.append(t)

    return np.array(inputs,np.float32)



