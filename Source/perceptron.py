#Masa Hu
#March 4, 2017
#Machine Learning
#
#                               Aritificial Neural Network Project
#############################################################################################################
#import necessary modules
#random for random wegiht assignment at perceptron creation
#os for system information
#time for making sure the program isn't running longer than 2 hours
#math for basic math functions
import random
import os
import time
import math
#############################################################################################################

#take account of program start time
start_time = time.time()

#create the Perceptron class with which we make perceptron objects out of
class Perceptron:
    #initialized variables are weights, bias, learning rate, and accuracy
    #inputs needed are
    #   learningRateIn
    #   length of the Input Layer
    def __init__(self,learningRateIn,lengthIn):
        self.lr = learningRateIn
        self.weights = []
        self.bias = -1
        self.accuracy = 0
        for i in range(lengthIn):
            x = float(random.randint(0,1))
            if x == 1:
                self.weights.append(-1)
            else:
                self.weights.append(1)

    #test the perceptron given inputs xs and weights ws
    #returns a single float value as output
    def test(self,xs,ws):
        sumxw = 0.0
        for x,w in zip(xs,ws):
            sumxw += float(x) * w
        sumxw += self.bias

        out = float(1 / (1 + pow(math.e,-1 * sumxw)))
        return out

    #learns what went wrong by applying the ANN algorithm
    #output is the sets:
    #   newWeights
    #   backPropagation
    #O stands for output of the perceptron from a test
    #preOs stands for the set of output of all perceptrons in the layer before it
    #weights are the set of weights that were used in the test
    #propIn is the set of propagation error that is being sent from the layer in front
    def learn(self,propIn,weights,preOs,O):
        newWeights = []
        backPropagation = []
        totalProp = 0.0

        for p,w in zip(propIn,weights):
            totalProp = totalProp + (p * w)
        for w,prex in zip(self.weights,preOs):
            sigma = totalProp * O * (1-O)
            totalError = sigma * float(prex)
            #THE -1 IS REQUIRED TO UPDATE IN THE DIRECTION OF THE MINIMUM!! https://en.wikipedia.org/wiki/Backpropagation
            newW = float(w) + self.lr * -1 * totalError

            propagation = sigma
            
            newWeights.append(newW)
            backPropagation.append(propagation)

        return newWeights,backPropagation

    #set function
    #sets this perceptrons weights with the new weights
    #output is null
    def setWeights(self,newWeights):
        for i,w in enumerate(newWeights):
            self.weights[i] = w

