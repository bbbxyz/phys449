"""
Convolutional Neural Network model

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


y_col = -2 #-3: temp, -2: energy, -1: magnetization
nbatch = 1 #number of batches for training
split_test = 0.4
learning_rate = 5e-7 #learning rate for gradient descent
epsilon = 0.05

#split for train/test
files = glob.glob('data/*.csv')
train, test = train_test_split(files,test_size=split_test)

feat = cst.lattice_size*cst.lattice_size
data_type = tf.float32

x = tf.placeholder(tf.float32, [None, feat])
y_ = tf.placeholder(tf.float32, [None,1])

#helper functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=0.0, stddev=0.2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

n_filters=[1, 4, 16]
filter_sizes=[3, 3, 3]

x_image = tf.reshape(x, [-1,cst.lattice_size,cst.lattice_size,1])
current_input = x_image

for layer_i, n_output in enumerate(n_filters[1:]):
    n_input = current_input.get_shape().as_list()[3]
    W = weight_variable([
            filter_sizes[layer_i],
            filter_sizes[layer_i],
            n_input, n_output])
    b = bias_variable([n_output])
    output = tf.nn.relu(
        tf.add(conv2d(current_input, W), b))
    current_input = output

conv_output_size = reduce(lambda x, y: x*y, current_input.get_shape().as_list() [1:])
conv_output_flat = tf.reshape(current_input, [-1, conv_output_size])

W_fc1 = weight_variable([conv_output_size, conv_output_size/4])
b_fc1 = bias_variable([conv_output_size/4])

h_fc1 = tf.nn.relu(tf.matmul(conv_output_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([conv_output_size/4, 1])
b_fc2 = bias_variable([1])

y=tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#define error as mean[(y-y')^2]
error = 0.5*tf.reduce_mean(tf.square(tf.subtract(y,y_)))

#create optimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)

diff = tf.square(y-y_)
accuracy = tf.sqrt(tf.reduce_mean(tf.cast(diff, tf.float32)))

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
print("Starting TensorFlow session")
sess = tf.Session()
sess.run(init)

def get_normalization_params():
    #find mean first
    sum = 0.0
    n=0
    for file in train:
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      sum += np.sum(Y)
      n += len(Y)
    mean = sum/float(n)
    #find stddev
    var=0
    for file in train:
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
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
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})


def test_set(testX, testY):
    score = sess.run(accuracy, feed_dict={x: testX , y_: testY,keep_prob: 1.0 })
    return score
    

 
#calculate accuracy on the testing set
def calculate_score():
    total_score = 0.0 
    for file in test:
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      X = df.values[:, :-3]

      #scale X to [0,1]
      X = (X+1.0)/2.0

      #Normalize y
      Y = (Y-mean)/float(stddev) 
      
      #split the dataset for training and testing
      total_score += test_set(X, Y)
    return total_score/float(len(test))


k = 0 #counter to keep track of number of times we've trained on the entire set 
sc = 1.0 
print("Calculation normalization parameters")
shuffle(train)
mean,stddev = get_normalization_params()
print("Done")

#start training until test set error is smaller than threshold 
print("%%%%%%%%%%%%%%%%%\nStarting training\n")
while(sc>epsilon): 
  shuffle(train)
  t0=time()
  for file in train:
      #print(file)
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      X = df.values[:, :-3]
    
      #scale X to [0,1]
      X = (X+1.0)/2.0
    
      #normalize Y
      Y = (Y-mean)/float(stddev)
      
      #split the dataset for training and testing
      train_set(X, Y)
  t1=time()    
  sc = calculate_score()
  t2=time()
  print("Error: %f, Training time: %is, Test time: %is,\
   iteration %i" % (sc, (t1-t0),(t2-t1), k),end='\n')
  k += 1
print("Done!")

print("\n")
print("Calculating validation score")
print(calculate_score())

#unnormalize if needed
#predictY = predictY*stddev+mean
#np.savetxt("testY.csv",testY , delimiter=",")
#np.savetxt("predictY.csv", predictY, delimiter=",")
