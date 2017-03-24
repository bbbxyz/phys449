"""
Convolutional Neural Network model for regression

"""
from __future__ import print_function
import Constants as cst
import glob, math, threading
import tensorflow as tf
import numpy as np
from random import shuffle
from time import time
from queue import Queue
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split


y_col = -2 #-3: temp, -2: energy, -1: magnetization
batch_size = 2 #number of samples from each file to sample when building batches
batch_sampling_size = 100
batch_per_epoch = 5
split_test = 0.005
learning_rate = 1e-2 #learning rate for gradient descent
epsilon = 0.05  #error at which to stop training
l2_alpha = 0.000
dim = cst.lattice_size
data_type = tf.float32

#split for train/test
files = glob.glob('data/*.csv')
train, test = train_test_split(files,test_size=split_test)

feat = cst.lattice_size*cst.lattice_size
data_type = tf.float32

x = tf.placeholder(tf.float32, [None, feat])
y_ = tf.placeholder(tf.float32, [None,1])

n_filters=      [ 4,  4]
filter_sizes=   [ 3,  3]
pool =          [ 0,  0]

#helper functions
def kernel_variable(shape):
  initial = tf.truncated_normal(shape, mean=0, stddev=0.1, dtype = data_type)
  return tf.Variable(initial)
  
def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=0, stddev=0.1, dtype = data_type)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype = data_type)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
x_image = tf.reshape(x, [-1,dim,dim,1])
current_input = x_image

for i, n_output in enumerate(n_filters):
    n_input = current_input.get_shape().as_list()[3]
    k = kernel_variable([
            filter_sizes[i],
            filter_sizes[i],
            n_input, n_output])
    b = bias_variable([n_output])
    output =  tf.nn.relu(tf.add(conv2d(current_input, k), b))
    if(pool[i]):
        output = avg_pool_2x2(output)
    current_input = tf.nn.relu(output)

conv_output_size = reduce(lambda x, y: x*y, current_input.get_shape().as_list() [1:])
conv_output_flat = tf.reshape(current_input, [-1, conv_output_size])
print(conv_output_size)
fc1size = conv_output_size
fc2size = conv_output_size/10

W_fc1 = weight_variable([conv_output_size, fc1size])
b_fc1 = bias_variable([fc1size])

W_fc2 = weight_variable([fc1size,fc2size ])
b_fc2 = bias_variable([fc2size])

W_o = weight_variable([fc2size, 1])
b_o= bias_variable([1])

keep_prob = tf.placeholder(data_type)
h_fc1 = tf.nn.relu(tf.matmul(conv_output_flat, W_fc1) + b_fc1)
h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
y = tf.matmul(h_fc2, W_o) + b_o

#l2_loss = l2_alpha*( tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_o) ) 
loss =  tf.reduce_mean(tf.square(y-y_))
accuracy = tf.reduce_mean(abs((y-y_)/y_)) 
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
print("Starting TensorFlow session")
sess = tf.Session()
sess.run(init)

def get_normalization_params():
    #find mean first
    sum = 0.0
    n=0
    for file in train[:100]:
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      sum += np.sum(Y)
      n += len(Y)
    mean = sum/float(n)
    #find stddev
    var=0
    for file in train[:100]:
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      var += np.sum(np.square(Y-mean))
    stddev = np.sqrt(var/float(n))
    return mean,stddev

def train_set(trainX, trainY):
    a, score = sess.run((train_step, accuracy),\
        feed_dict={x: trainX, y_: trainY, keep_prob: 0.5})
    return score

def test_set(testX, testY):
    score, lo, predic, real = sess.run((accuracy,loss,  y, y_), feed_dict={x: testX , y_: testY, keep_prob: 1.0 })
    print(predic[-1],real[-1])
    #print(lo)
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
      total_score += test_set(X, Y)
      
    return total_score/float(len(test))

threads = []

def get_batch():
    

def create_batches():
    minibatchX = np.array([]).reshape(0,feat)
    minibatchY = np.array([]).reshape(0,1)
    sel_f = np.random.random_integers(0,len(train)-1,batch_sampling_size).tolist()
    samples = [train[i] for i in sel_f]
    for file in samples:
      #print(file)
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      X = df.values[:, :-3]
      sel = np.random.random_integers(0,len(Y)-1, batch_size)
      
      #scale X to [0,1]
      X = (X[sel]+1.0)/2.0
      minibatchX = np.vstack((minibatchX,X))
      
      #normalize Y
      Y = (Y[sel]-mean)/float(stddev)
      minibatchY = np.vstack((minibatchY,Y))

k = 0 #counter to keep track of number of times we've trained on the entire set 
sc = 1.0 
print("Calculating normalization parameters")
#shuffle(train)
mean,stddev = get_normalization_params()
print("Done")
#start training until test set error is smaller than threshold 
print("Starting training\n")
while(sc>epsilon): 
    t0=time()
    for j in range(batch_per_epoch):
      minibatchX, minibatchY = get_batch()
      train_err = train_set(minibatchX, minibatchY)
    t2=time()    
    sc = calculate_score()
    t3=time()
    print(train_err)
    print("Error: %f, Training time: %is, Test time: %is,\
    Epoch %i" % (sc, (t2-t0), (t3-t2), k),end='\n')
    k += 1
    
print("Done!")
print("Calculating validation score")
print(calculate_score())

#unnormalize if needed
#predictY = predictY*stddev+mean
#np.savetxt("testY.csv",testY , delimiter=",")
#np.savetxt("predictY.csv", predictY, delimiter=",")
