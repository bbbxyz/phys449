"""
Convolutional Neural Network model for regression
"""

from __future__ import print_function
import Constants as cst
import glob, math, threading
from multiprocessing import Process, Queue
import tensorflow as tf
import numpy as np
from random import shuffle
from time import time
#from queue import Queue
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

data_directory='data/*.csv'  #data directory 
y_col = -2                   #-3: temp, -2: energy, -1: magnetization
batch_size = 200            #number of samples to take from each file
split_test = 0.5             #test/train split
learning_rate = 1e-6        #learning rate for gradient descent
epsilon = 0.01               #error at which to stop training (UNUSED)
l2_alpha = 0.00              #regularization term
max_epoch = 1          #how many epochs to run
dim = cst.lattice_size       #lattice dimensions, change if running on old data

#conv. layers parameters
n_filters=      [16, 32]
filter_sizes=   [ 3,  3]
pool =          [ 1,  1]
fc1_size = 1000

#these parameters shouldn't change unless we run out of memory
data_type = tf.float32
batches = Queue(maxsize=100)
test_batches = Queue(maxsize=30)

#split for train/test
files = glob.glob(data_directory)
train, test = train_test_split(files,test_size=split_test)


#helper functions
def kernel_variable(shape):
  initial = tf.truncated_normal(shape, mean=0, stddev=0.05, dtype = data_type)
  return tf.Variable(initial)
  
def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=0, stddev=1/float(shape[1]), dtype = data_type)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape, dtype = data_type)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

feat = cst.lattice_size*cst.lattice_size
x = tf.placeholder(tf.float32, [None, feat])
y_ = tf.placeholder(tf.float32, [None,1])
                       
x_image = tf.reshape(x, [-1,dim,dim,1])
current_input = x_image

for i, n_output in enumerate(n_filters):
    n_input = current_input.get_shape().as_list()[3]
    k = kernel_variable([
            filter_sizes[i],
            filter_sizes[i],
            n_input, n_output])
    b = bias_variable([n_output])
    output =  tf.add(conv2d(current_input, k), b)
    if(pool[i]):
        output = max_pool_2x2(output)
    current_input = tf.tanh(output)

conv_output_size = reduce(lambda x, y: x*y, current_input.get_shape().as_list() [1:])
b_conv = bias_variable([conv_output_size])
conv_output_flat = tf.reshape(current_input, [-1, conv_output_size])\
                    + b_conv
print(conv_output_size)
W_fc1 = weight_variable([conv_output_size, fc1_size])
b_fc1 = bias_variable([fc1_size])

W_o = weight_variable([fc1_size, 1])
b_o= bias_variable([1])

keep_prob = tf.placeholder(data_type)
fc1 = tf.tanh(tf.matmul(conv_output_flat, W_fc1) + b_fc1)
y = tf.matmul(fc1, W_o) + b_o

l2_loss = l2_alpha*(tf.nn.l2_loss(W_o) + tf.nn.l2_loss(W_fc1)) 
loss = l2_loss + tf.reduce_mean(tf.square(y-y_))
accuracy = tf.reduce_mean(abs((y-y_)/y_)) 
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
print("Starting TensorFlow session")
sess = tf.Session()
sess.run(init)

def get_normalization_params():
    '''
    Calculates the (approximate) mean and standard deviation of the training set
    
    Returns mean, standard_deviation
    '''
    sum = 0.0
    n=0
    for file in train[:10]:
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      sum += np.sum(Y)
      n += len(Y)
    mean = sum/float(n)
    var=0
    for file in train[:10]:
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      var += np.sum(np.square(Y-mean))
    stddev = np.sqrt(var/float(n))
    return mean,stddev

def train_set(trainX, trainY):
    '''
    Performs training on the training batch given
    
    Returns training accuracy
    '''
    a, acc = sess.run((train_step, accuracy),\
        feed_dict={x: trainX, y_: trainY, keep_prob: 0.5})
    return acc

def test_set(testX, testY):
    '''
    Performs testing on the batch given
    
    Returns testing accuracy and predictions
    '''
    score= sess.run(accuracy, feed_dict={x: testX , y_: testY, keep_prob: 1.0 })
    return score
 
def predict_set(predX, predY):
  return sess.run(y, feed_dict={x: predX , y_: predY, keep_prob: 1.0 })
 
def calculate_score(complete):
    '''
    Calculates the accuracy on the entire testing set if
    complete is set to True.
    Else calculates it on the first 20 elements
    Returns testing accuracy
    '''
    total_score = 0.0 
    if(complete):
      size = len(test)
      for file in test:
        df=pd.read_csv(file)
        Y = df.values[:, y_col]
        Y = np.reshape(Y, (len(Y),1))
        X = df.values[:, :-3] 
        Y = (Y-mean)/float(stddev)
        score = test_set(X, Y) 
        total_score += score 
        
    else:
      size=20
      for j in range(size):
        X, Y = get_test_batch()
        score = test_set(X, Y) 
        total_score += score 
    
    return total_score/float(size)

def plot_predictions():
    '''
    Plots predictions and actual data for 5 files randomly chosen
     from the test set
    '''
    for file in test[:5]:
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      X = df.values[:, :-3]

      #Normalize y
      Y = (Y-mean)/float(stddev)
      predict = predict_set(X, Y) 
      plt.plot(Y*stddev + mean, 'k')
      plt.plot(predict*stddev + mean, 'r--')        
    plt.show()

def plot_confusion():
    '''
    Creates a confusion plot for the testing set
    X axis = real data
    Y axis = predictions
    '''
    for file in test:
      df=pd.read_csv(file)
      Y = df.values[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      X = df.values[:, :-3]
      
      #Normalize y
      Y = (Y-mean)/float(stddev)
      predict = predict_set(X, Y) 
      plt.scatter(Y,predict, c='k', marker='.', s=1)
    plt.plot([-5,5],[-5,5],'r')
    plt.show()

def get_batch():
    '''
    Gets and returns a batch from the training queue
    '''
    batch = batches.get()
    return batch[0],batch[1]

def get_test_batch():
    '''
    Gets and returns a batch from the testing queue
    '''
    batch = test_batches.get()
    return batch[0],batch[1]

def create_batches():
    '''
    Creates batches from the training set and places them on the queue
    Runs as long as flag_running is set to 1
    '''
    while(flag_running):
      #shuffle(train)
      for file in train:
        if(flag_running):
          df=pd.read_csv(file)
          Y = df.values[:, y_col]
          Y = np.reshape(Y, (len(Y),1))
          X = df.values[:, :-3]
          sel = np.random.random_integers(0,len(Y)-1, batch_size)
          X = X[sel]
          Y = (Y[sel]-mean)/float(stddev)
          if(not batches.full()):
            batches.put((X,Y))

def create_test_batches():
    '''
    Creates batches from the testing set and places them on the queue
    Runs as long as flag_running is set to 1
    '''
    while(flag_running):
        for file in test:
          if(flag_running):
            df=pd.read_csv(file)
            Y = df.values[-batch_size:, y_col]
            Y = np.reshape(Y, (len(Y),1))
            X = df.values[-batch_size:, :-3]
            Y = (Y-mean)/float(stddev)
            if(not test_batches.full()):
              test_batches.put((X,Y))

def train_dataset():
    '''
    Runs the training loop until max number of epochs is reached.
    '''
    k = 0 #counter to keep track of number of times we've trained on the entire set 
    sc = 1.0 
    best = 10.0
    frac = int(len(train)/10)
    #frac = 2
    while(k<max_epoch): 
        train_err = 0.0
        t0=time()
        for j in range(frac):
          batchX, batchY = get_batch()
          train_err += train_set(batchX[0:], batchY[0:])
        t2=time()    
        test_err = calculate_score(False)
        t3=time()
        train_err = train_err/float(frac)
        if(best>=test_err):
            best= test_err
            #save the model if test error is lower
            save_path = saver.save(sess, "saved/CNN.ckpt")
        print("Train/Test Error: %f/%f, Train/Test Time: %is/%is\
        Epoch %i" % (train_err, test_err, (t2-t0), (t3-t2), k),end='\n')
        k += 1


print("Calculating normalization parameters")
mean,stddev = get_normalization_params()
print("Creating threads")
flag_running = 1
creator_thread1 = Process(target=create_batches, daemon=True)
creator_thread2 = Process(target=create_test_batches, daemon=True)
trainer_thread= threading.Thread(target=train_dataset, daemon=True)

print("Starting training")
creator_thread1.start()
creator_thread2.start()
trainer_thread.start()
trainer_thread.join()
flag_running = 0
creator_thread1.join()
creator_thread2.join()
print("Training Completed")

saver.restore(sess, "saved/CNN.ckpt")
print("Model restored")
print("Calculating validation score")
print(calculate_score(True))
plot_predictions()
plot_confusion()
