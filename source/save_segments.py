#!/usr/local/bin/python

import segmentModule
import sys
import os

CARDBOARD_DIR = "/home/masa/Projects/TrashClassification_capstone/source/categories/cardboard/ingroup"
TREEMATTER_DIR = "/home/masa/Projects/TrashClassification_capstone/source/categories/treematter/ingroup"
PLYWOOD_DIR = "/home/masa/Projects/TrashClassification_capstone/source/categories/plywood/ingroup"
CONSTRUCTION_DIR = "/home/masa/Projects/TrashClassification_capstone/source/categories/construction_waste/ingroup"
OUT_DIR = "segments/"

def test_saveSegments():

    cat1_list = os.listdir(CARDBOARD_DIR)
    cat2_list = os.listdir(TREEMATTER_DIR)
    cat3_list = os.listdir(PLYWOOD_DIR)
    cat4_list = os.listdir(CONSTRUCTION_DIR)
    for f1,f2,f3,f4 in zip(cat1_list,cat2_list,cat3_list,cat4_list):
        full_dir1 = CARDBOARD_DIR + f1
        full_dir2 = TREEMATTER_DIR + f2
        full_dir3 = PLYWOOD_DIR + f3
        full_dir4 = CONSTRUCTION_DIR + f4

        image1 = segmentModule.normalizeImage(full_dir1)
        image2 = segmentModule.normalizeImage(full_dir2)
        image3 = segmentModule.normalizeImage(full_dir3)
        image4 = segmentModule.normalizeImage(full_dir4)

        original, labels = segmentModule.getSegments(image1,False)
        segmentModule.saveSegments(original,labels,False,OUT_DIR,f1)

        original, labels = segmentModule.getSegments(image2,False)
        segmentModule.saveSegments(original,labels,False,OUT_DIR,f2)

        original, labels = segmentModule.getSegments(image3,False)
        segmentModule.saveSegments(original,labels,False,OUT_DIR,f3)

        original, labels = segmentModule.getSegments(image4,False)
        segmentModule.saveSegments(original,labels,False,OUT_DIR,f4)


