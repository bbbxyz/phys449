"""
Convolutional Neural Network model for regression

"""

import argparse
import glob, math, threading, sys, os
import tensorflow as tf
import numpy as np
from functools import reduce

import Constants as cst

#helper functions
def kernel_variable(shape):
  initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
  return tf.Variable(initial)
  
def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=0, stddev=0.5/float(shape[1]))
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


class CNN_model:
    
    def __init__(self, n_layers, dim, mean, stddev):
    
        #conv. layers parameters
        self.n_layers = n_layers
        self.n_filters=[16, 32,  64, 128, 256, 512, 1024][:n_layers]
        self.filter_sizes=   [ 1,  1,   1,   1,   1,    1,    1][:n_layers]*filter_size        
        
    def start_session():
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        return self.sess
        
        
    x = tf.placeholder(tf.float32, [None, dim*dim])
    y_ = tf.placeholder(tf.float32, [None,1])
    x_image = tf.reshape(x, [-1,dim,dim,1])*0.5+0.5
    current_input = x_image

    #create conv. layers with maxpool and relu
    for i, n_output in enumerate(self.n_filters):
        n_input = current_input.get_shape().as_list()[3]
        k = kernel_variable([
                self.filter_sizes[i],
                self.filter_sizes[i],
                n_input, n_output])
        b = bias_variable([n_output])
        output =  max_pool(tf.add(conv2d(current_input, k), b))
        current_input = tf.nn.relu(output)
        
    #calculate size of last conv. layer
    conv_output_size = reduce(lambda x, y: x*y, current_input.get_shape().as_list() [1:])
    b_conv = bias_variable([conv_output_size])

    #flatten last conv.layer and apply dropout
    keep_prob = tf.placeholder(data_type)
    conv_output_flat = tf.nn.dropout(tf.reshape(current_input, [-1, conv_output_size])\
                        + b_conv, keep_prob)

    #create two fully-connected layers
    W_fc1 = weight_variable([conv_output_size, fc1_size])
    b_fc1 = bias_variable([fc1_size])
    W_o = 10*weight_variable([fc1_size, 1])
    b_o= 10*bias_variable([1])
    fc1 = tf.nn.relu(tf.matmul(conv_output_flat, W_fc1) + b_fc1)
    y = tf.matmul(fc1, W_o) + b_o

    #squared-error loss function
    loss =  tf.reduce_mean(tf.square(y-y_))

    #define error as (y-y')/y'
    outy = (y*stddev)+mean
    outy_ =(y_*stddev)+mean
    error = abs(tf.reduce_mean(abs(outy-outy_))/tf.reduce_mean(outy_))

    train_step = tf.train.RMSPropOptimizer(learning_rate = learning_rate,\
            momentum = 0.1, decay=0.5).minimize(loss)

    self.saver = tf.train.Saver()
    if not os.path.exists("saved"):
        os.makedirs("saved")
        
    def train_set(trainX, trainY):
        '''
        Performs training on the training batch given
        
        Returns training accuracy
        '''
        a, score = sess.run((train_step, accuracy),\
            feed_dict={x: trainX, y_: trainY, keep_prob: 0.5})
        return score

    def test_set(testX, testY):
        '''
        Performs testing on the batch given
        
        Returns testing accuracy and predictions
        '''
        score , pred= sess.run((accuracy, y), feed_dict={x: testX , y_: testY, keep_prob: 1.0 })
        return score
     
    def save_model():
        save_path = saver.save(sess, "saved/CNN.ckpt")
        
    def restore_model():
        saver.restore(sess, "saved/CNN.ckpt")
        
