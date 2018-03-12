import featureReader as fr
import numpy as np

if __name__ == '__main__':

    inputs,labels = fr.getSegmentBatch(1000)
    print(inputs.shape)
    print(labels.shape)

