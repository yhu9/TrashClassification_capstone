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


#######################################################################################
#######################################################################################
#Main Function

def main(unused_argv):

    if len(sys.argv) > 2 and sys.argv[1] == 'train':

        #read the training set
        train_x, train_y,test_x,test_y = featureReader.read_training_txt(sys.argv[2])
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

    elif len(sys.argv) > 2 and sys.argv[1] == 'test':

        img = featureReader.cnn_readOneImg(sys.argv[2])

        #shrink image because my poor laptop can't handle it otherwise
        smaller = featureReader.shrinkImg(img)
        eval_single_img(smaller,sys.argv[3])
    else:
        print("unexpected arguments")


if __name__ == "__main__":
    tf.app.run()

