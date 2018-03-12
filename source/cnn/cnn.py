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
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

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
#https://stackoverflow.com/questions/37649060/tensorflow-restoring-a-graph-and-model-then-running-evaluation-on-a-single-imag
#adapted code for testing a single image using the learned model and returning a segmented image using cnn

#######################################################################################
#######################################################################################
#Main Function

def main(unused_argv):

#start off by checking the number of arguments so we don't immediatly crash the program
#must have at least the mode and image for this to run
#arg[0] is program name
#arg[1] is mode of operation {train, test, see}
#arg[2] is the directory of the image to read
    if(len(sys.argv) >= 3):

        #image to be read in
        featureReader.read_training_img(sys.argv[3])

        #the number of features output from the last convolution layer to the first fully connected layer
        magic_number = int(constants.IMG_WIDTH * constants.IMG_HEIGHT * constants.CNN_LAYER1)

        #our CNN architecture
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
            predictions = tf.matmul(fullyConnected,weights['out'])+biases['out']

        layers = [conv1,activation1,fullyConnected,output]

        #Make predictions and calculate loss and accuracy given the inputs and labels
        #keep track of the layers to see how they were trained
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
        #requires the mode of operation and the image to train on
        if sys.argv[1] =='train':

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

        elif len(sys.argv) == 3 and sys.argv[1] == 'test':

            img = featureReader.cnn_readOneImg(sys.argv[2])

            #shrink image because my poor laptop can't handle it otherwise
            smaller = featureReader.shrinkImg(img)
            eval_single_img(smaller,sys.argv[3])

        elif(sys.argv[1] == 'see'):
            ckpt_dir = sys.argv[2]
            def split_into_rgb_channels(image):
                '''Split the target image into its red, green and blue channels.
                image - a numpy array of shape (rows, columns, 3).
                output - three numpy arrays of shape (rows, columns) and dtype same as
                           image, containing the corresponding channels.
                '''
                red = image[:,:,2]
                green = image[:,:,1]
                blue = image[:,:,0]
                return red, green, blue

            #restore the graph and make the predictions and show the segmented image
            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                vals = weights['W_conv1'].eval()
                i = 1
                for k in range(vals.shape[3]):
                    extracted_filter = vals[:,:,:,k]
                    r, g, b = split_into_rgb_channels(extracted_filter)

                    cv2.namedWindow('filter r',cv2.WINDOW_NORMAL)
                    cv2.namedWindow('filter g',cv2.WINDOW_NORMAL)
                    cv2.namedWindow('filter b',cv2.WINDOW_NORMAL)
                    cv2.namedWindow('filter rgb',cv2.WINDOW_NORMAL)
                    cv2.imshow('filter r',r)
                    cv2.imshow('filter g',g)
                    cv2.imshow('filter b',b)
                    cv2.imshow('filter rgb',extracted_filter)

                    f_out = "filter_" + str(i)
                    i += 1
                    cv2.imwrite(f_out,extracted_filter)
                    cv2.waitKey(0)

        elif(sys.argv[1] == 'test' and len(sys.argv) == 4):
            ckpt_dir = sys.argv[3]

            #restore the graph and make the predictions and show the segmented image
            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                inputs = []
                for i in range(len(img)):
                    if(i >= 14 and i < len(img) - 14):
                        inputs.append([])
                        for j in range(len(img[0])):
                            if(j >= 14 and j < len(img[0]) - 14):
                                box = img[i-14:i+14,j-14:j+14]
                                prediction = predict_op.eval({x:[box]})
                                if(prediction == [0]):
                                    inputs[len(inputs) - 1].append(0)
                                elif(prediction == [1]):
                                    inputs[len(inputs) - 1].append(1)
                                elif(prediction == [2]):
                                    inputs[len(inputs) - 1].append(2)
                                elif(prediction == [3]):
                                    inputs[len(inputs) - 1].append(3)
                    print('%i out of %i' % (i,len(img)))

                best_guess = np.array(inputs)
                canvas = np.zeros((len(best_guess),len(best_guess[0]),3))

                canvas[best_guess == 0] = [0,0,255]
                canvas[best_guess == 1] = [0,255,0]
                canvas[best_guess == 2] = [255,0,0]
                canvas[best_guess == 3] = [0,255,255]

                cv2.imshow('original',img)
                cv2.namedWindow('original',cv2.WINDOW_NORMAL)
                cv2.imshow('segmented image',canvas)
                cv2.namedWindow('segmented image',cv2.WINDOW_NORMAL)

                cv2.imwrite('segmentation.png',canvas)

                cv2.waitKey(0)
    else:
        print("oopsies")
        print("need (train,img_dir) or (test,img_dir,model)")


if __name__ == "__main__":
    tf.app.run()

