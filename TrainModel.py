"""
Script for training the model for regression.

Creates 2 subprocesses that fetch training/testing datafiles, splits  them 
in batches and places batches on queues. 

Saves trained model in folder 'saved/'

Usage: $ python TrainModel.py [number of layers] [n. epochs]  [data directory] 

"""

from __future__ import print_function
import glob, threading, sys
from multiprocessing import Process, Queue
import numpy as np
from random import shuffle
from time import time
from sklearn.model_selection import train_test_split

import Constants as cst
import CNNModel as mdl

batch_size = 200            
learning_rate = 1e-3

def get_normalization_params():
    '''
    Calculates the (approximate) mean and standard deviation of the training set
    
    Returns mean, standard_deviation
    '''
    sum = 0.0
    n=0
    for file in train[:5]:
      df = np.loadtxt(file, delimiter=',')
      Y = df[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      sum += np.sum(Y)
      n += len(Y)
    mean = sum/float(n)
    var=0
    for file in train[:5]:
      df=np.loadtxt(file, delimiter=',')
      Y = df[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      var += np.sum(np.square(Y-mean))
    stddev = np.sqrt(var/float(n))
    frac = int(len(Y)/float(batch_size))
    return mean,stddev,frac

def calculate_score(model, complete):
    '''
    Calculates the accuracy on the entire testing set if
    complete is set to True.
    Else calculates it on the first 20 elements
    Returns testing accuracy
    '''
    total_score = 0.0 
    if(complete):
      size = len(test)*frac
    else:
      size = len(test)/5*frac
      
    for j in range(size):
        print("Testing Batch %i out of %i                              " % (j+1, size),end='\r')
        batchX, batchY = get_test_batch()
        score = model.test_set(batchX, batchY)
        total_score += score
    return total_score/float(size)


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

def create_batches(batches, train, mean, stddev, y_col):
    '''
    Creates batches from the training set and places them on the queue
    
    '''
    while(True):
      shuffle(train)
      for file in train:
          df=np.loadtxt(file, delimiter=',')
          Y = df[:, y_col]
          Y = np.reshape(Y, (len(Y),1))
          X = df[:, :-3]
          Y = (Y-mean)/float(stddev)
          for i in range(int(len(Y)/batch_size)):
            batchX = X[i*batch_size:(i+1)*batch_size,:]
            batchY = Y[i*batch_size:(i+1)*batch_size]
            batches.put((batchX,batchY))

def create_test_batches(test_batches, test, mean, stddev, y_col):
    '''
    Creates batches from the testing set and places them on the queue
    
    '''
    while(True):
        for file in test:
            df=np.loadtxt(file, delimiter=',')
            Y = df[:, y_col]
            Y = np.reshape(Y, (len(Y),1))
            X = df[:, :-3]
            Y = (Y-mean)/float(stddev)
            for i in range(int(len(Y)/batch_size)):
                batchX = X[i*batch_size:(i+1)*batch_size,:]
                batchY = Y[i*batch_size:(i+1)*batch_size]
                test_batches.put((batchX,batchY))
            
def train_dataset(model, max_epoch):
    '''
    Runs the training loop until max number of epochs is reached.
    '''
    k = 0 
    best = 1e10
    nbatch = int(len(train)*frac)
    while(k<max_epoch): 
        train_err = 0.0
        t0=time()
        for j in range(nbatch):
          batchX, batchY = get_batch()
          er = model.train_set(batchX, batchY)
          train_err += er
          print("Training Batch %i out of %i. Minibatch error:%.2f" % (j+1, nbatch, er),end='\r')

        t2=time()    
        test_err = calculate_score(model, True)
        t3=time()
        train_err = train_err/float(nbatch)
        if(best>=test_err):
            best= test_err
            #save the model if test error is lower
            model.save_model()        
        print("Train/Test Error: %f/%f, Train/Test Time: %is/%is\
        Epoch %i" % (train_err, test_err, (t2-t0), (t3-t2), k),end='\n')
        k += 1

if __name__ == '__main__':
        
    if(len(sys.argv) < 4):
        print("Not enough input arguments")
        print("Correct usage: $ python TrainModel.py [number of layers] [n. epochs]  [data directory]")
        exit(1)
    n_layers = int(sys.argv[1])  #get number of layers from command line
    max_epoch = int(sys.argv[2])        #how many epochs to run
    data_directory=sys.argv[3] + '*.csv'  #data directory 
    
    dim = cst.lattice_size
    y_col = -2                   #-3: temp, -2: energy, -1: magnetization
    split_test = 0.3            #test/train split
    queue_size = 2            #decrease this if you run out of memory

    #split for train/test
    files = glob.glob(data_directory)
    train, test = train_test_split(files,test_size=split_test)
    
    batches = Queue(maxsize=queue_size*len(train))
    test_batches = Queue(maxsize=queue_size*len(test))

    print("Calculating normalization parameters")
    mean,stddev, frac = get_normalization_params()
    
    model = mdl.CNN_model(n_layers, dim, \
        mean, stddev, learning_rate)
    model.start_session()
    
    print("Creating threads")
    creator_thread1 = Process(target=create_batches,\
            args=(batches, train, mean, stddev, y_col,), daemon=True)
    creator_thread2 = Process(target=create_test_batches,\
            args=(test_batches, test, mean, stddev, y_col,), daemon=True)
    trainer_thread= threading.Thread(target=train_dataset, args=(model,max_epoch,), daemon=True)

    print("Starting training")
    creator_thread1.start()
    creator_thread2.start()
    trainer_thread.start()
    trainer_thread.join()
    creator_thread1.terminate()
    print("Training Completed")

    print("Restoring best model")
    model.restore_model()
    
    print("Calculating validation score")
    print(calculate_score(model, True))
    creator_thread2.terminate()
     
