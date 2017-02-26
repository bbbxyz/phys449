"""
@author: mass
convolution model based on Tensorflow's deep MNIST model
https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/

todo:
- optimize hyperparams
"""
from __future__ import print_function

import Constants as cst
import glob, math
import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
from time import time
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split



nbatch = cst.iterations/100 #number of batches for training
split_test = 0.4
learning_rate = 5e-6 #learning rate for gradient descent
epsilon = 0.1

#split for train/test
files = glob.glob('data/*.csv')
train, test = train_test_split(files,test_size=split_test)

feat = cst.lattice_size*cst.lattice_size
data_type = tf.float32

x = tf.placeholder(tf.float32, [None, feat])
y_ = tf.placeholder(tf.float32, [None,1])

#helper functions
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

feat_conv1=16
feat_conv2=8
size_fc=32

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

y=tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#define error as mean[(y-y')^2]
error = 0.5*tf.reduce_mean(tf.square(tf.subtract(y,y_)))

#create optimizer
opt = tf.train.AdamOptimizer(learning_rate)
grads = opt.compute_gradients(error)
train_step = opt.minimize(error)

diff = tf.square(y-y_)
accuracy = tf.sqrt(tf.reduce_mean(tf.cast(diff, tf.float32)))

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
#sess = tf.Session(config=tf.ConfigProto(
#    intra_op_parallelism_threads=4))
sess = tf.Session()
sess.run(init)

def get_normalization_params():
    #find mean first
    sum = 0.0
    n=0
    for file in train:
      df=pd.read_csv(file)
      Y = df.values[:, -1]
      Y = np.reshape(Y, (len(Y),1))
      sum += np.sum(Y)
      n += len(Y)
    mean = sum/float(n)
    #find stddev
    var=0
    for file in train:
      df=pd.read_csv(file)
      Y = df.values[:, -1]
      Y = np.reshape(Y, (len(Y),1))
      var += np.sum(np.square(Y-mean))
    stddev = np.sqrt(var/float(n))
    return mean,stddev
    
def train_set(trainX, trainY):
    for i in range(nbatch):
      lowbound = int(len(trainX)*i/nbatch)
      upbound = int(len(trainX)*(i+1)/nbatch)
      batch_xs =  trainX[lowbound:upbound, :]
      batch_ys =  trainY[lowbound:upbound]  
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})


def test_set(testX, testY):
    score = sess.run(accuracy, feed_dict={x: testX , y_: testY,keep_prob: 1.0 })
    return score
    

 
#calculate accuracy on the testing set
def calculate_score():
    total_score = 0.0 
    for file in test:
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
      total_score += test_set(X, Y)
    return total_score/float(len(test))


k = 0 #counter to keep track of number of times we've trained on the entire set 
sc = 1.0 
mean,stddev = get_normalization_params()
print(mean,stddev)

#start training until test set error is smaller than threshold 
print("%%%%%%%%%%%%%%%%%\nStarting training\n%%%%%%%%%%%%%%%%%\n")
while(sc>epsilon): 
  shuffle(train)
  t0=time()
  for file in train:
      #print(file)
      df=pd.read_csv(file)
      Y = df.values[:, -1]
      Y = np.reshape(Y, (len(Y),1))
      X = df.values[:, :-1]
    
      #scale X to [0,1]
      X = (X+1.0)/2.0
    
      #normalize Y
      Y = (Y-mean)/stddev
      
      #split the dataset for training and testing
      train_set(X, Y)
  t1=time()    
  sc = calculate_score()
  t2=time()
  print("Error: %f, Training time: %is, Test time: %is,\
   iteration %i" % (sc, (t1-t0),(t2-t1), k),end='\n')
  k += 1

print("\n")
print(calculate_score())

#unnormalize if needed
#predictY = predictY*stddev+mean
#np.savetxt("testY.csv",testY , delimiter=",")
#np.savetxt("predictY.csv", predictY, delimiter=",")
