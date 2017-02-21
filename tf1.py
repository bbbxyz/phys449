# -*- coding: utf-8 -*-
"""
@author: mass
"""
import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split


t0=time()
nbatch = 100 #number of batches for training
split_test=0.01
learning_rate = 0.005
N=20


#feat = X.shape[1]
feat = N*N

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
layer1 = tf.nn.tanh( tf.matmul(x, w1) + b1 )
layer2 = tf.nn.tanh( tf.matmul(layer1, w2) + b2)
y = tf.matmul(layer2, wout) + bout

#define error as sum[(y-y')^2]
error = tf.reduce_sum(tf.square(tf.subtract(y, y_)))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)

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
	  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	  print(sess.run(error, feed_dict={x: batch_xs , y_: batch_ys }))
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
	#X = (X+1.0)/2.0

	#scale Y values to range [0,1]
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
score = sess.run(accuracy, feed_dict={x: testX , y_: testY })
predictY = sess.run(y, feed_dict={x: testX , y_: testY })
print(score)
#unnormalize if needed
#predictY = predictY*(maxY-minY)+minY
print( predictY[0:10], testY[0:10])
#np.savetxt("testY.csv",testY , delimiter=",")
#np.savetxt("predictY.csv", predictY, delimiter=",")
		
