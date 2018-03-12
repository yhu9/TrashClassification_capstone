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

####################################################################################################################################
#######################################################################################
#######################################################################################
#Main Function

def main(unused_argv):

    #check the number of arguments given with running the program
    #must be at least two
    #argv[1] is the mode of operation {test,see,train}
    #argv[2] is the input image
    #argv[3] is the optional
    if len(sys.argv) >= 2:

        #################################################################################################################
        #################################################################################################################
        #Define our Convolutionary Neural Network from scratch
        x = tf.placeholder('float',[None,28,28,3])
        y = tf.placeholder('float',[None,constants.CNN_CLASSES])
        weights = {}
        biases = {}

        #first convolution layers
        weights['W_conv1'] = tf.Variable(tf.random_normal([5,5,3,constants.CNN_LAYER1]))
        biases['b_conv1'] = tf.Variable(tf.random_normal([constants.CNN_LAYER1]))
        conv1 = tf.nn.conv2d(x,weights['W_conv1'],strides=[1,1,1,1],padding='SAME',name='conv1')
        activation1 = tf.nn.relu(conv1 + biases['b_conv1'])

        #create our first fully connected layer
        #magic number = width * height * n_convout
        magic_number = int(constants.IMG_WIDTH * constants.IMG_HEIGHT * constants.CNN_LAYER1)
        with tf.name_scope('Fully_Connected_1'):
            with tf.name_scope('activation'):
                weights['W_fc'] = tf.Variable(tf.random_normal([magic_number,constants.CNN_FULL1]))
                biases['b_fc'] = tf.Variable(tf.random_normal([constants.CNN_FULL1]))
                layer1_input = tf.reshape(activation1,[-1,magic_number])
                fullyConnected = tf.nn.relu(tf.matmul(layer1_input,weights['W_fc']) + biases['b_fc'])
                #fullyConnected = tf.nn.dropout(fullyConnected,constants.KEEP_RATE)
            tf.summary.histogram('activations_3',fullyConnected)

        #Final fully connected layer for classification
        with tf.name_scope('output'):
            weights['out'] = tf.Variable(tf.random_normal([constants.CNN_FULL1,constants.CNN_CLASSES]))
            biases['out'] = tf.Variable(tf.random_normal([constants.CNN_CLASSES]))
            predictions = tf.matmul(fullyConnected,weights['out'])+biases['out']

        #define optimization and accuracy creation
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=y))
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))
            correct_prediction = tf.cast(correct_prediction,tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

        predict_op = tf.argmax(predictions,1)
        tf.summary.scalar('accuracy',accuracy)

        #################################################################################################################
        #################################################################################################################
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        #training mode trained on the image
        if(sys.argv[1] == 'train'):
            #read the image

            #Run the session/CNN and train/record accuracies at given steps
            #with tf.Session() as sess:
            #net = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
                sess.run(init)
                merged = tf.summary.merge_all()

                for epoch in range(constants.CNN_EPOCHS):
                    #get a random image
                    img = featureReader.getStitchedImage()

                    #get an image batch
                    batch_x,batch_y = featureReader.getImgBatch(img,1000)
                    eval_x,eval_y = featureReader.getImgBatch(img,1000)

                    optimizer.run(feed_dict={x: batch_x, y: batch_y})

                    #sess.run([merged,optimizer],feed_dict={x: batch_x, y: batch_y})
                    if epoch % 1 == 0:
                        acc = accuracy.eval({x: eval_x, y: eval_y})
                        print('epoch: ' + str(epoch) + '     ' +
                                'accuracy: ' + str(acc))

                #saver
                save_path = saver.save(sess,'./model/cnn2_model.ckpt')
                print("Model saved in file: %s" % save_path)

        elif(sys.argv[1] == 'see' and len(sys.argv) == 3):
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

                    cv2.waitKey(0)
                    #f_out = "./results/filter_" + str(i)
                    #i += 1
                    #cv2.imwrite(f_out,extracted_filter)

        #testing method needs a saved check point directory (model)
        elif(sys.argv[1] == 'test' and len(sys.argv) == 4):
            #get the directory of the checkpoint
            ckpt_dir = sys.argv[3]

            #read the image
            img = featureReader.cnn_readOneImg(sys.argv[2])

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

                #create the segmented image
                canvas[best_guess == 0] = [0,0,255]
                canvas[best_guess == 1] = [0,255,0]
                canvas[best_guess == 2] = [255,0,0]
                canvas[best_guess == 3] = [0,255,255]

                #show the original image and the segmented image and then save the results
                cv2.imshow('original',img)
                cv2.namedWindow('original',cv2.WINDOW_NORMAL)
                cv2.imshow('segmented image',canvas)
                cv2.namedWindow('segmented image',cv2.WINDOW_NORMAL)
                cv2.imwrite('segmentation.png',canvas)
                cv2.waitKey(0)

                #count the percentage of each category
                cat1_count = np.count_nonzero(best_guess == 0)
                cat2_count = np.count_nonzero(best_guess == 1)
                cat3_count = np.count_nonzero(best_guess == 2)
                cat4_count = np.count_nonzero(best_guess == 3)
                total = cat1_count + cat2_count + cat3_count + cat4_count

                #get the percentage of each category
                p1 = cat1_count / total
                p2 = cat2_count / total
                p3 = cat3_count / total
                p4 = cat4_count / total

                #print out to the terminal what the percentage of each category was
                print("%s : %f.2" % (constants.CAT1,p1))
                print("%s : %f.2" % (constants.CAT2,p2))
                print("%s : %f.2" % (constants.CAT3,p3))
                print("%s : %f.2" % (constants.CAT4,p4))


                greatest = max(cat1_count,cat2_count,cat3_count,cat4_count)

                #print out to the terminal what the most common category was for the image
                if(greatest == cat1_count):
                    print("the most common category is: " + constants.CAT1)
                elif(greatest == cat2_count):
                    print("the most common category is: " + constants.CAT2)
                elif(greatest == cat3_count):
                    print("the most common category is: " + constants.CAT3)
                elif(greatest == cat4_count):
                    print("the most common category is: " + constants.CAT4)
                else:
                    print("sorry something went wrong counting the predictions")

    else:
        print("oopsies")
        print("argv[1]: mode of operation (test,train,see)")
        print("argv[2]: image direcory to train/test on")
        print("argv[3]: *OPTIONAL* model directory if testing")
        print("need (train,img_dir) or (test,img_dir,model)")

if __name__ == "__main__":
    tf.app.run()
