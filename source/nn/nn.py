#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for TBI, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import math
import constants
import random
import sys
import re
import os

#import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

###################################################################

###################################################################
#1. Convolutional layer
#2. Pooling layers
#3. Convolutional layer
#4. pooling layer
#5. Fully connected layer
#6. Logits layer
###################################################################

####################################################################################################################################
#Helper Functions
####################################################################################################################################


#We try to read all the files all at once. Maybe I should fix that...
#inputs are 1d extracted feature vectors
#labels are one hot
#returns inputs and labels
def read_training_txt():
    inputs = []
    labels = []
    for f in sys.argv[3:]:
        if len(f) == 0:
            print("bad files passed")
            print("expected: ")
            print("file_directory_in received: " + f)
            sys.exit()

        group = re.findall("(?<=resources/)[a-z]+",f)
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
            inputs.append(img)

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
#5x5, 11x11, 25x25 surrounding area of each pixel used for training
#all training images must be passed when calling nn.py
def cnn_readOneImg(image_dir):
    inputs = []
    labels = []

    img = cv2.imread(image_dir,cv2.IMREAD_COLOR)

    inputs.append(img)

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

#Some monitoring basics on the tensorflow website
#https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#numbe

####################################################################################################################################
#Define our Convolutionary Neural Network from scratch

#Training function for our convolution neural network
def train_convolution_network(x,y,training_set,training_labels,testing_set,testing_labels):

    #the number of features output from the last convolution layer to the first fully connected layer
    w = math.sqrt(len(training_set[0]))
    h = w
    magic_number = int(w * h * constants.CNN_LAYER1)

    #our CNN architecture
    def CNN(x):
            weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,constants.CNN_LAYER1])),
                       #'W_conv2':tf.Variable(tf.random_normal([5,5,constants.CNN_LAYER1,constants.CNN_LAYER2])),
                       'W_fc':tf.Variable(tf.random_normal([magic_number,constants.N_FEAT_FULL1])),
                       'out':tf.Variable(tf.random_normal([constants.N_FEAT_FULL1, constants.N_CLASSES]))}

            biases = {'b_conv1':tf.Variable(tf.random_normal([constants.CNN_LAYER1])),
                       #'b_conv2':tf.Variable(tf.random_normal([constants.N_FEAT_LAYER2])),
                       'b_fc':tf.Variable(tf.random_normal([constants.N_FEAT_FULL1])),
                       'out':tf.Variable(tf.random_normal([constants.N_CLASSES]))}

            #reshape the input
            area = len(training_set[0])
            w = math.sqrt(area)
            h = w
            features = tf.reshape(x,[-1,int(w),int(h),1])

            #create our 1st convolutionary layer with histogram summaries of activations
            conv1 = tf.nn.conv2d(features,weights['W_conv1'],strides=[1,1,1,1],padding='SAME',name='conv1')
            activation1 = tf.nn.relu(conv1 + biases['b_conv1'])

            #create our first fully connected layer
            with tf.name_scope('Fully_Connected_1'):
                with tf.name_scope('activation'):
                    layer1_input = tf.reshape(activation1,[-1,magic_number])
                    fullyConnected = tf.nn.relu(tf.matmul(layer1_input,weights['W_fc']) + biases['b_fc'])
                tf.summary.histogram('activations_3',fullyConnected)

            #Final fully connected layer for classification
            with tf.name_scope('output'):
                output = tf.matmul(fullyConnected,weights['out'])+biases['out']

            layers = [conv1,activation1,fullyConnected,output]

            return layers,output

    #Make predictions and calculate loss and accuracy given the inputs and labels
    #keep track of the layers to see how they were trained
    layers, predictions = CNN(x)

    #define optimization and accuracy creation
    with tf.name_scope('cost'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=y))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))
        correct_prediction = tf.cast(correct_prediction,tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    tf.summary.scalar('accuracy',accuracy)

    #Run the session/CNN and train/record accuracies at given steps
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter(constants.LOG_DIR + '/train',sess.graph)
        #test_writer = tf.summary.FileWriter(constants.LOG_DIR + '/test')

        for epoch in range(constants.CNN_EPOCHS):

            batch_x,batch_y = mnist.train.next_batch(50)
            optimizer.run(feed_dict={x: batch_x, y: batch_y})

            #sess.run([merged,optimizer],feed_dict={x: batch_x, y: batch_y})

            if epoch % 1 == 0:
                acc = accuracy.eval({x: testing_set, y: testing_labels})
                print('epoch: ' + str(epoch) + '     ' +
                        'accuracy: ' + str(acc)
                        )

        #saver
        saver = tf.train.Saver()
        save_path = saver.save(sess,'./model/trained_nn_model')
        print("Model saved in file: %s" % save_path)

#Training function for our neural network
def train_neural_network(x,y,training_set,training_labels,testing_set,testing_labels):

    #define our neural network architecture
    def NN(x):
        weights = {
                'W_layer1':tf.Variable(tf.random_normal([len(training_set[0]),constants.NN_HIDDEN1])),
                'W_layer2':tf.Variable(tf.random_normal([constants.NN_HIDDEN1,constants.NN_HIDDEN2])),
                'W_out':tf.Variable(tf.random_normal([constants.NN_HIDDEN2,constants.NN_CLASSES]))
            }

        biases = {
                'B_layer1':tf.Variable(tf.random_normal([constants.NN_HIDDEN1])),
                'B_layer2':tf.Variable(tf.random_normal([constants.NN_HIDDEN2])),
                'B_out':tf.Variable(tf.random_normal([constants.NN_CLASSES]))
            }

        layer_1 = tf.add(tf.matmul(x,weights['W_layer1']),biases['B_layer1'])
        out_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(out_1,weights['W_layer2']),biases['B_layer2'])
        out_2 = tf.nn.relu(layer_2)
        out_layer = tf.add(tf.matmul(out_2,weights['W_out']),biases['B_out'])

        layers = [layer_1,layer_2]

        return layers,out_layer


    #create the NN architecture
    layers, predictions = NN(x)

    #define optimization and accuracy creation
    with tf.name_scope('cost'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=y))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(constants.NN_LEARNING_RATE).minimize(cost)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))
        correct_prediction = tf.cast(correct_prediction,tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    tf.summary.scalar('accuracy',accuracy)

    #Run the session/CNN and either train or record accuracies at given steps
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter(constants.LOG_DIR + '/train',sess.graph)
        #test_writer = tf.summary.FileWriter(constants.LOG_DIR + '/test')

        for epoch in range(constants.NN_EPOCHS):

            #batch_x,batch_y = mnist.train.next_batch(100)
            #define your input and labels here

            #make the batches
            low = -1; high = -1; bad_batchsize = False
            if(constants.NN_BATCHSIZE > len(training_set) - 1):
                bad_batchsize = True

            goodrange = len(training_set) - constants.NN_BATCHSIZE - 1
            rand_index = random.randint(0,goodrange)
            if((rand_index + constants.NN_BATCHSIZE) < len(training_set)):
                low = rand_index
                high = rand_index + constants.NN_BATCHSIZE
            elif((rand_index - constants.NN_BATCHSIZE) >= 0):
                low = rand_index - constants.NN_BATCHSIZE
                high = rand_index
            else:
                bad_batchsize = True

            #run the batches if all is good
            batch_x = training_set[low:high]
            batch_y = training_labels[low:high]
            sess.run([merged,optimizer],feed_dict={x: batch_x, y: batch_y})

            #get the testing accuracy
            if epoch % 1 == 0:
                acc = accuracy.eval({x: testing_set, y: testing_labels})
                print('epoch: ' + str(epoch) + '     ' +
                        'accuracy: ' + str(acc)
                        )

        #saver
        saver = tf.train.Saver()
        save_path = saver.save(sess,'./model/trained_nn_model')
        print("Model saved in file: %s" % save_path)


def testmodel():

    saver = tf.train.Saver()

    #Run the session/CNN and either train or record accuracies at given steps
    sess = tf.InteractiveSession()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(constants.LOG_DIR + '/train',sess.graph)
    test_writer = tf.summary.FileWriter(constants.LOG_DIR + '/test')
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,"./resources/trained_nn_model")

    #define your input and labels here
    if epoch % 1 == 0:
        acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
        print('epoch: ' + str(epoch) + '     ' +
                'accuracy: ' + str(acc)
                )

#######################################################################################
#######################################################################################
#Main Function

def main(unused_argv):

    if len(sys.argv) > 2 and sys.argv[1] == 'train' and sys.argv[2] == 'nn':

        #read the training set
        train_x, train_y,test_x,test_y = read_training()
        #train_x = mnist.train.images
        #train_y = mnist.train.labels
        #test_x = mnist.test.images
        #test_y = mnist.test.labels

        #initialize placeholders for input and output of neural network
        x = tf.placeholder('float',[None,len(train_x[0])])
        y = tf.placeholder('float',[None,constants.NN_CLASSES])
        #y = tf.placeholder('float',[None,10])

        #train the network
        train_neural_network(x,y,train_x,train_y,train_x,train_y)

    elif len(sys.argv) > 2 and sys.argv[1] =='train' and sys.argv[2] == 'cnn':

        train_x = mnist.train.images
        train_y = mnist.train.labels
        test_x = mnist.test.images
        test_y = mnist.test.labels

        #initialize placeholders for input and output of neural network
        w = math.sqrt(len(train_x[0]))
        h = w

        x = tf.placeholder('float',[None,len(train_x[0])])
        y = tf.placeholder('float',[None,constants.N_CLASSES])

        train_convolution_network(x,y,train_x,train_y,test_x,test_y)
    elif len(sys.argv) > 2 and sys.argv[1] == 'test':

        read_unknown()
        print("working on it")
        #print(type(mnist.train.images))
        #print((mnist.train.labels[0]))
    else:
        print("unexpected arguments")


if __name__ == "__main__":
    tf.app.run()

