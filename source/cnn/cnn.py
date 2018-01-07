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
import cv2
import math
import sys

#Python Modules
import constants
import featureReader

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

####################################################################################################################################
#Define our Convolutionary Neural Network from scratch

#Training function for our convolution neural network
def train_cnn2(x,y,img):
    #the number of features output from the last convolution layer to the first fully connected layer

    #our CNN architecture
    def CNN(x):

        weights = {}
        biases = {}

        feature = tf.reshape(x,[-1,28,28,3])

        #first convolution layers
        weights['W_conv1'] = tf.Variable(tf.random_normal([5,5,3,6]))
        biases['b_conv1'] = tf.Variable(tf.random_normal([6]))
        conv1 = tf.nn.conv2d(feature,weights['W_conv1'],strides=[1,1,1,1],padding='SAME',name='conv1')
        activation1 = tf.nn.relu(conv1 + biases['b_conv1'])

        #create our first fully connected layer
        magic_number = int(28 * 28 *6)
        with tf.name_scope('Fully_Connected_1'):
            with tf.name_scope('activation'):
                weights['W_fc'] = tf.Variable(tf.random_normal([magic_number,1024]))
                biases['b_fc'] = tf.Variable(tf.random_normal([1024]))
                layer1_input = tf.reshape(activation1,[-1,magic_number])
                fullyConnected = tf.nn.relu(tf.matmul(layer1_input,weights['W_fc']) + biases['b_fc'])
            tf.summary.histogram('activations_3',fullyConnected)

        #Final fully connected layer for classification
        with tf.name_scope('output'):
            weights['out'] = tf.Variable(tf.random_normal([1024,4]))
            biases['out'] = tf.Variable(tf.random_normal([4]))
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

            batch_x,batch_y = featureReader.getImgBatch(img,10000)
            eval_x,eval_y = featureReader.getImgBatch(img,1000)

            optimizer.run(feed_dict={x: batch_x, y: batch_y})

            #sess.run([merged,optimizer],feed_dict={x: batch_x, y: batch_y})

            if epoch % 1 == 0:
                acc = accuracy.eval({x: eval_x, y: eval_y})
                print('epoch: ' + str(epoch) + '     ' +
                        'accuracy: ' + str(acc)
                        )

        #saver
        saver = tf.train.Saver()
        save_path = saver.save(sess,'./model/cnn2/trained_cnn2_model')
        print("Model saved in file: %s" % save_path)

#Training function for our convolution neural network
def train_cnn(x,y,training_set,training_labels,testing_set,testing_labels):

    #the number of features output from the last convolution layer to the first fully connected layer
    w = math.sqrt(len(training_set[0]))
    h = w
    magic_number = int(w * h * constants.CNN_LAYER1)

    #our CNN architecture
    def CNN(x):
            weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,constants.CNN_LAYER1])),
                       'W_conv2':tf.Variable(tf.random_normal([5,5,constants.CNN_LAYER1,constants.CNN_LAYER2])),
                       'W_fc':tf.Variable(tf.random_normal([magic_number,constants.N_FEAT_FULL1])),
                       'out':tf.Variable(tf.random_normal([constants.N_FEAT_FULL1, constants.N_CLASSES]))}

            biases = {'b_conv1':tf.Variable(tf.random_normal([constants.CNN_LAYER1])),
                       'b_conv2':tf.Variable(tf.random_normal([constants.N_FEAT_LAYER2])),
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
        save_path = saver.save(sess,'./model/cnn/trained_cnn_model')
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
        save_path = saver.save(sess,'./model/nn/trained_nn_model')
        print("Model saved in file: %s" % save_path)



#https://stackoverflow.com/questions/37649060/tensorflow-restoring-a-graph-and-model-then-running-evaluation-on-a-single-imag
#adapted code for testing a single image using the learned model and returning a segmented image using cnn
def eval_single_img(img,ckpt_dir):

    # below code adapted from @RyanSepassi, however not functional
    # among other errors, saver throws an error that there are no
    # variables to save

    #our CNN architecture
    def CNN(x):

        weights = {}
        biases = {}

        #first convolution layers
        weights['W_conv1'] = tf.Variable(tf.random_normal([5,5,3,6]))
        biases['b_conv1'] = tf.Variable(tf.random_normal([6]))
        conv1 = tf.nn.conv2d(x,weights['W_conv1'],strides=[1,1,1,1],padding='SAME',name='conv1')
        activation1 = tf.nn.relu(conv1 + biases['b_conv1'])

        #create our first fully connected layer
        magic_number = int(28 * 28 *6)
        with tf.name_scope('Fully_Connected_1'):
            with tf.name_scope('activation'):
                weights['W_fc'] = tf.Variable(tf.random_normal([magic_number,1024]))
                biases['b_fc'] = tf.Variable(tf.random_normal([1024]))
                layer1_input = tf.reshape(activation1,[-1,magic_number])
                fullyConnected = tf.nn.relu(tf.matmul(layer1_input,weights['W_fc']) + biases['b_fc'])
            tf.summary.histogram('activations_3',fullyConnected)

        #Final fully connected layer for classification
        with tf.name_scope('output'):
            weights['out'] = tf.Variable(tf.random_normal([1024,4]))
            biases['out'] = tf.Variable(tf.random_normal([4]))
            output = tf.matmul(fullyConnected,weights['out'])+biases['out']

        layers = [conv1,activation1,fullyConnected,output]

        return layers,output

    #keep track of the layers to see how they were trained
    x = tf.placeholder('float',[None,28,28,3])
    y = tf.placeholder('float',[None,4])
    layers, predictions = CNN(x)

    #define optimization and accuracy creation
    with tf.name_scope('cost'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=y))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(constants.NN_LEARNING_RATE).minimize(cost)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))
        correct_prediction = tf.cast(correct_prediction,tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    predict_op = tf.argmax(predictions,1)

    #reset the graph
    #tf.reset_default_graph()
    saver = tf.train.Saver()

    #predict every pixel of the image we cut off 14 pixels off the sides (28 / 2)
    inputs = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(i >= 14 and i < len(img) - 14 and j >= 14 and j < len(img[0]) - 14):
                box = img[i-14:i+14,j-14:j+14]
                inputs.append(box)
                if(j == len(img[0]) - 1):
                    print("%f complete" % (float(i) / float(len(img))))

    #restore the graph and make the predictions and show the segmented image
    with tf.Session() as sess:
        saver.restore(sess,ckpt_dir)
        print("session restored!")

        weights = tf.get_variable("W_conv1",shape=[5,5,3,6])
        print(weights.eval())

        #tx,ty = featureReader.getImgBatch(img,100)
        #acc = accuracy.eval({x: tx, y: ty})
        #feed_dict = {x: [inputs]}
        #best_guess = sess.run(predict_op,feed_dict)

        #canvas = np.array(best_guess)
        #canvas[best_guess == [1,0,0,0]] = [0,0,255]
        #canvas[best_guess == [0,1,0,0]] = [0,255,0]
        #canvas[best_guess == [0,0,1,0]] = [255,0,0]
        #canvas[best_guess == [0,0,0,1]] = [255,255,0]

        #cv2.imshow('segmented Image',canvas)
        #cv2.waitKey(0)

#######################################################################################
#######################################################################################
#Main Function

def main(unused_argv):

    if len(sys.argv) > 2 and sys.argv[1] == 'train' and sys.argv[2] == 'nn':

        #read the training set
        train_x, train_y,test_x,test_y = featureReader.read_training_txt(sys.argv[3])
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

        if(len(sys.argv) == 4):
            featureReader.read_training_img(sys.argv[3])
            train_x = mnist.train.images
            train_y = mnist.train.labels
            test_x = mnist.test.images
            test_y = mnist.test.labels

            #initialize placeholders for input and output of neural network
            w = math.sqrt(len(train_x[0]))
            h = w

            x = tf.placeholder('float',[None,len(train_x[0])])
            y = tf.placeholder('float',[None,constants.N_CLASSES])

            train_cnn(x,y,train_x,train_y,test_x,test_y)
        else:
            print("error with args")

    elif len(sys.argv) > 2 and sys.argv[1] =='train' and sys.argv[2] == 'cnn2':

        img = featureReader.cnn_readOneImg(sys.argv[3])
        train_x,train_y = featureReader.getImgBatch(img,1000)
        test_x,test_y = featureReader.getImgBatch(img,200)

        x = tf.placeholder('float',[None,28,28,3])
        y = tf.placeholder('float',[None,4])

        train_cnn2(x,y,img)

        #train_convolution_network(x,y,train_x,train_y,test_x,test_y)
    elif len(sys.argv) > 2 and sys.argv[1] == 'test':

        img = featureReader.cnn_readOneImg(sys.argv[2])

        #shrink image because my poor laptop can't handle it otherwise
        smaller = featureReader.shrinkImg(img)
        eval_single_img(smaller,sys.argv[3])
    else:
        print("unexpected arguments")


if __name__ == "__main__":
    tf.app.run()

