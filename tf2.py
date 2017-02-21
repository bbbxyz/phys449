# -*- coding: utf-8 -*-
"""
@author: mass
convolution model based on Tensorflow's deep MNIST model
https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/

todo:
- load test set by batches to avoid running out of memory
- optimize hyperparams
- 
"""

import Constants as cst
import glob, math
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split



nbatch = 100 #number of batches for training
split_test=0.001
learning_rate = 5e-5
NUM_CHANNELS=1 #number of channels in the input "image"

feat = cst.lattice_size*cst.lattice_size
data_type = tf.float32

#this is where we build the neural net
x = tf.placeholder(tf.float32, [None, feat])
y_ = tf.placeholder(tf.float32, [None,1])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

feat_conv1=8
feat_conv2=16
size_fc=1024

W_conv1 = weight_variable([3, 3, 1, feat_conv1])
b_conv1 = bias_variable([feat_conv1])

x_image = tf.reshape(x, [-1,cst.lattice_size,cst.lattice_size,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, feat_conv1, feat_conv2])
b_conv2 = bias_variable([feat_conv2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([cst.lattice_size/4 * cst.lattice_size/4 * feat_conv2, size_fc])
b_fc1 = bias_variable([size_fc])

h_pool2_flat = tf.reshape(h_pool2, [-1, cst.lattice_size/4 * cst.lattice_size/4 *feat_conv2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([size_fc, 1])
b_fc2 = bias_variable([1])

y=tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#define error as mean[(y-y')^2]
error = tf.reduce_mean(tf.square(tf.subtract(y,y_)))

opt = tf.train.GradientDescentOptimizer(learning_rate)
grads =opt.compute_gradients(error)
train_step = opt.minimize(error)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def train_set(trainX, trainY):
    for i in range(nbatch):
      lowbound = int(len(trainX)*i/nbatch)
      upbound = int(len(trainX)*(i+1)/nbatch)
      batch_xs =  trainX[lowbound:upbound, :]
      batch_ys =  trainY[lowbound:upbound]  
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8})
      if(i%(nbatch/5) ==0):
		err = sess.run(error, feed_dict={x: batch_xs , y_: batch_ys,keep_prob: 0.8 })
		print(math.sqrt(err))
    
#load all our datasets
files = glob.glob('data/*.csv')
datasets =[]
testYs=()
testXs=()
for file in files:
    print(file)
    df=pd.read_csv(file)
    Y = df.values[:, -1]
    Y = np.reshape(Y, (len(Y),1))
    X = df.values[:, :-1]
    
    #scale X to [0,1]
    X = (X+1.0)/2.0
    
    #scale Y values to range 0,1]
    maxY = float(max(Y))
    minY = float(min(Y))
    Y = (Y-minY)/(maxY-minY) 
        
    #split the dataset for training and testing
    trainX, testX, trainY, testY = train_test_split(X,Y,test_size=split_test)
    testYs = testYs + (testY,)
    testXs = testXs + (testX,)
    train_set(trainX, trainY)

testX = np.vstack(testXs)
testY = np.vstack(testYs)

#calculate accuracy on the testing set
diff = tf.square(y-y_)
accuracy = tf.sqrt(tf.reduce_mean(tf.cast(diff, tf.float32)))/tf.reduce_mean(y_)
score = sess.run(accuracy, feed_dict={x: testX , y_: testY,keep_prob: 1.0 })
predictY = sess.run(y, feed_dict={x: testX , y_: testY, keep_prob: 1.0 })
print(score)

#unnormalize if needed
#predictY = predictY*(maxY-minY)+minY
print( predictY[0:10], testY[0:10], (testY[0:10]-predictY[0:10]))
#np.savetxt("testY.csv",testY , delimiter=",")
#np.savetxt("predictY.csv", predictY, delimiter=",")
