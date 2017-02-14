# -*- coding: utf-8 -*-
"""
@author: mass
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
filename = "300.csv"
t0=time()
nbatch = 1000 #number of batches for training
split_test=0.2
learning_rate = 1e-5

#load the data
df = pd.read_csv(filename)
Y = df.values[:, -1]
Y = np.reshape(Y, (len(Y),1))
X = df.values[:, :-1]

#scale X to [0,1]
X = (X+1.0)/2.0

#scale Y values to range [0,1]
maxY = max(Y)
minY = min(Y)
Y = (Y-minY)/(maxY-minY)

#split the dataset for training and testing
trainX, testX, trainY, testY = train_test_split(X,Y,test_size=split_test)
feat = X.shape[1]


#this is where we build the neural net
#for now it is only a linear regression model
n_hidden1 = 64
n_hidden2 = 32

x = tf.placeholder(tf.float32, [None, feat])
y_ = tf.placeholder(tf.float32, [None,1])

#initialize weights and biases for every layer
w1 = tf.Variable(tf.random_normal([feat, n_hidden1]))
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
wout =tf.Variable(tf.random_normal([n_hidden2, 1]))

b1 = tf.Variable(tf.random_normal([n_hidden1]))
b2 = tf.Variable(tf.random_normal([n_hidden2]))
bout =tf.Variable(tf.random_normal([1]))

#create each hidden layer and the output layer 
layer1 = tf.nn.sigmoid( tf.matmul(x, w1) + b1 )
layer2 = tf.nn.sigmoid ( tf.matmul(layer1, w2) + b2)
y = tf.matmul(layer2, wout) + bout

#define error as sum[(y-y')^2]
error = tf.reduce_sum(tf.square(y- y_), reduction_indices=[1])
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()

#perform gradient descent
t1=time()
sess.run(init)
for i in range(nbatch):
  lowbound = int(len(trainX)*i/nbatch)
  upbound = int(len(trainX)*(i+1)/nbatch)
  batch_xs =  trainX[lowbound:upbound, :]
  batch_ys =  trainY[lowbound:upbound]  
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
t2 =time()
print(t2-t1,t1-t0)

#calculate accuracy on the testing set
diff = tf.square(y-y_)
accuracy = tf.sqrt(tf.reduce_mean(tf.cast(diff, tf.float32)))
print(sess.run(accuracy, feed_dict={x: testX , y_: testY }))
predictY = sess.run(y, feed_dict={x: testX , y_: testY })

#unnormalize if needed
#predictY = predictY*(maxY-minY)+minY
print( predictY, testY)
