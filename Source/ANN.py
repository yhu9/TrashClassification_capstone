#Masa Hu
#2/17/2017
#ANN Project 1
#Machine Learning
#############################################################################################################################################
from perceptron import Perceptron
import sys
import math
import re
import numpy as np
from matplotlib import pyplot as plt
#############################################################################################################################################
#ANN.py takes the following arguments
#1. file directory of training set
#2. file directory of validation set
#3. learning rate (as a float)
#4. convergence Condition (as a float)
#
#The convergence condition is measured as the difference in the validation accuracy after 100 epochs. Convergence is met with the following condition
#
#           convergence {   1   if V_n - V_n-100 <= condition           V_n = accuracy current     V_n-100 = accuracy initial
#                       {   0   if V_n - V_n-100 > condition            condition = convergence condition   V_i is an accuracy output during an epoch
#
#It is recommended that the convergence condition be set to a value between [0.01,0.2] as accuracies are in the range [0,1]
#
#
#Equation for testing
#
#   
#
#Equation for learning
#
#   
#############################################################################################################################################
#CONSTANTS
OUT_CATEGORIES = 4



#Start Class ANN
#confusion matrix is [actual][expected]
#Actual = row
#expected = col
class ANN:
    def __init__(self,learningRateIn,improvementRateIn,filename):
        self.lr = learningRateIn
        self.convergence = improvementRateIn
        self.hiddenlayer = []
        self.outputlayer = []
        self.fileout = filename
        self.confusionMatrix = np.zeros(shape=(OUT_CATEGORIES,OUT_CATEGORIES))

    #basic set methods for each layer
    def setHiddenLayer(self,hiddenLayerIn):
        self.hiddenlayer = hiddenLayerIn

    def setOutputLayer(self,outputLayerIn):
        self.outputlayer = outputLayerIn

    #takes a list of numbers and outputs a list with soft max as 1 and all other values as 0
    def softMax(self,listIn):
        output = []
        m = max(listIn)
        index = listIn.index(m)
        for i,out in enumerate(listIn):
            if i == index:
                output.append(1)
            else:
                output.append(0)

        return output

    #creates a list of 26 0s, and sets the index given as 1
    def expectedOutput(self,index):
        output = [0] * 26
        output[index] = 1

        return output

    #executes the ANN with the given set hiddenlayer and outputlayer.
    #testInstances and validationInstances will have to be provided as input
    #tInstances = [[],[],...,[]]
    #vInstances = [[],[],...,[]]
    #the first number of an instance is the expected output
    def execute(self,tInstances,vInstances):
        if len(self.hiddenlayer) == 0 or len(self.outputlayer) == 0:
            print "error, please set hidden and output layers"
            return -1
        #Initiate some variables
        self.confusionMatrix = np.zeros(shape=(26,26))
        converge = False
        epochNum = 1
        accuracyT = []
        accuracyV = []
        totalT = len(tInstances)
        totalV = len(vInstances)
        while converge == False:
            correctT,correctV = 0.0,0.0

            #See how well our Initial ANN does on the validation set without any learning.
            #Output is stored in the fileout
            if epochNum == 1:
                initial = 0
                for i in vInstances:
                    #Test the testset
                    expected = self.expectedOutput(int(float(i[0])))
                    output1 = []
                    for p in self.hiddenlayer:
                        x = p.test(i[1:],p.weights)
                        output1.append(x)
                    output2 = []
                    for p in self.outputlayer:
                        x = p.test(output1,p.weights)
                        output2.append(x)

                    soft = self.softMax(output2)
                    if expected == soft:
                        initial += 1
                
                accuracyInitial = float(initial)/float(totalV)
                print("Initial accuracy validset: %f, epochNum: %f" % (accuracyInitial,0))
                with open(self.fileout,'a') as fout:
                    fout.write("Initial accuracy of validset: %f, epochNum: %f" % (accuracyInitial,0))
                    fout.write('\n')

            #Start learning on each instance in tInstances
            #learning function and testing function is part of the perceptron module
            for i in tInstances:
                #Test an instance
                output1 = []
                for p in self.hiddenlayer:
                    x = p.test(i[1:],p.weights)
                    output1.append(x)
                output2 = []
                for p in self.outputlayer:
                    x = p.test(output1,p.weights)
                    output2.append(x)

                soft = self.softMax(output2)
                expected = self.expectedOutput(int(float(i[0])))
                
                #Learn from an instance
                if expected != soft:
                    #Output Layer
                    backProp2 = []
                    newWeights2 = []
                    for p,x,e in zip(self.outputlayer,output2,expected):
                        propagation = -1 * (e - x)
                        newW,backProp= p.learn([propagation],[1],output1,x)
                        newWeights2.append(newW)
                        backProp2.append(backProp)
                    #Hidden Layer
                    backProp1 = []
                    newWeights1 = []
                    for index,p,x in zip(range(len(self.hiddenlayer)),self.hiddenlayer,output1):
                        propweights = []
                        propagations = []
                        for ptmp,proptemp in zip(self.outputlayer,backProp2):
                            propweights.append(ptmp.weights[index])
                            propagations.append(proptemp[index])
                        newW,backProp = p.learn(propagations,propweights,i[1:],x)
                        
                        newWeights1.append(newW)
                        backProp1.append(backProp)
                    
                    #update weights
                    for index,w in zip(range(len(self.hiddenlayer)),newWeights1):
                        self.hiddenlayer[index].setWeights(w)
                    for index,w in zip(range(len(self.outputlayer)),newWeights2):
                        self.outputlayer[index].setWeights(w)
                else:
                    correctT += 1
                
            #Test on the Validation set
            #no learning is done during this part, only validation
            #confusion matrix is updated here
            self.confusionMatrix = np.zeros(shape=(26,26))
            for i in vInstances:
                #Test the validationset
                output1 = []
                for p in self.hiddenlayer:
                    x = p.test(i[1:],p.weights)
                    output1.append(x)
                output2 = []
                for p in self.outputlayer:
                    x = p.test(output1,p.weights)
                    output2.append(x)
                expected = self.expectedOutput(int(float(i[0])))
                soft = self.softMax(output2)
                
                #We update the confusion matrix and make sure it doesn't go greater than 255
                if expected == soft:
                    index = expected.index(1)
                    self.confusionMatrix[index][index] += 1
                    correctV += 1
                else:
                    indexE = expected.index(1)
                    indexA = soft.index(1)
                    self.confusionMatrix[indexE][indexA] += 1
                
            #Write all the pertinant info of an epoch to an output file.
            accuracyT.append(float(correctT)/float(totalT))
            accuracyV.append(float(correctV)/float(totalV))
            print("testset: %f, validset: %f, epochNum: %f" % (accuracyT[-1],accuracyV[-1],epochNum))
            with open(self.fileout,'a') as fout:
                fout.write("testset: %f, validset: %f, epochNum: %f" % (accuracyT[-1],accuracyV[-1],epochNum))
                fout.write('\n')
            if (epochNum % 100) == 0:
                i = epochNum - 100
                diff = (accuracyV[epochNum - 1] - accuracyV[i])
                print self.convergence
                if diff <= self.convergence:
                    converge = True
                self.confusionMatrix[self.confusionMatrix > 255] = 255
                imgplot = plt.imshow(self.confusionMatrix, cmap='gray')
                plt.ylabel("actual")
                plt.xlabel("expected")
                plt.savefig("images/lr"+"confusionmatrix" + str(epochNum) + "lr"+str(self.lr)+".png")
                #plt.show()
            if epochNum == 1000:
                converge = True
            if epochNum == 1:
                self.confusionMatrix[self.confusionMatrix > 255] = 255
                imgplot = plt.imshow(self.confusionMatrix, cmap='gray')
                plt.ylabel("actual")
                plt.xlabel("expected")
                plt.savefig("images/lr"+"confusionmatrix" + str(epochNum) + "lr"+str(self.lr)+".png")
                #plt.show()
            
            epochNum += 1
        #end while
    #end def execute
#end class ANN

#####################################################################################################################################
#main
if len(sys.argv) >= 5: 
    testsetfile = str(sys.argv[1])
    validsetfile = str(sys.argv[2])
    learningrate = float(sys.argv[3])
    convergenceCond = float(sys.argv[4])
    fileout = str(sys.argv[5])

    testInstances = []
    validationInstances = []
    hiddenlayer = []
    outputlayer = []
    output = []

    #Parse and read the testSet and the validationSet in order to make the testInstances and the
    #validation instances that needs to be passed to the ANN
    with open(testsetfile,'r') as ftest, open(validsetfile,'r') as fvalid:
        tlines = ftest.read().splitlines()
        vlines = fvalid.read().splitlines()
        for line in tlines:
            tokens = re.findall('\d+\.?\d+',line)
            testInstances.append(tokens)

        for line in vlines:
            tokens = re.findall('\d+\.?\d+',line)
            validationInstances.append(tokens)
    
    #preset the number of layers and number of perceptrons in each layer. No perceptron is made for the input layer by creating 
    #all the perceptrons
    #hlayernum = len(validationInstances)/(5*(number of outputs at output layer + number of inputs into a node at inputlayer))
    hlayernum = 40
    for x in range(hlayernum):
        p = Perceptron(learningrate,len(validationInstances[0][1:]))
        hiddenlayer.append(p)
    print("hidden layer created: %i" % len(hiddenlayer))
    for x in range(26):
        p = Perceptron(learningrate,len(hiddenlayer))
        outputlayer.append(p)
    print("output layer created: %i" % len(outputlayer))


    #create and setup the ANN
    #then execute with the test_instances and the validation_instances
    ArtificialNeuralNetwork = ANN(learningrate,convergenceCond,fileout)
    print("ANN created with")
    print("learning rate: %f, convergenceCond: %f" % (learningrate,convergenceCond))
    ArtificialNeuralNetwork.setHiddenLayer(hiddenlayer)
    ArtificialNeuralNetwork.setOutputLayer(outputlayer)
    ArtificialNeuralNetwork.execute(testInstances,validationInstances)

    #Execute stops at either the following conditions
    #1. 500 epochs
    #2. convergenceCond is met
    #3. something goes terribly wrong

else:
    print "run the program using the following parameters"
    print "testset validset learningrate convergencecond"








