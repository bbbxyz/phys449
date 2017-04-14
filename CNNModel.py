"""
Convolutional Neural Network model for regression

"""
import os
import tensorflow as tf
from functools import reduce

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
    
    def __init__(self, n_layers, dim, mean, stddev, lr):
    
        #conv. layers parameters
        self.dim = dim
        self.learning_rate = lr
        self.n_layers = n_layers
        self.n_filters=[16, 32,  64, 128, 256, 512, 1024][:n_layers]
        self.data_type = tf.float32
        
        self.x = tf.placeholder(tf.float32, [None, self.dim*self.dim])
        self.y_ = tf.placeholder(tf.float32, [None,1])
        x_image = tf.reshape(self.x, [-1,self.dim,self.dim,1])*0.5+0.5
        current_input = x_image

        #create conv. layers with maxpool and relu
        for i, n_output in enumerate(self.n_filters):
            n_input = current_input.get_shape().as_list()[3]
            k = kernel_variable([
                    3,3,
                    n_input, n_output])
            b = bias_variable([n_output])
            output =  max_pool_2x2(tf.add(conv2d(current_input, k), b))
            current_input = tf.nn.relu(output)
            
        #calculate size of last conv. layer
        conv_output_size = reduce(lambda x, y: x*y, current_input.get_shape().as_list() [1:])
        b_conv = bias_variable([conv_output_size])

        #flatten last conv.layer and apply dropout
        self.keep_prob = tf.placeholder(self.data_type)
        conv_output_flat = tf.nn.dropout(tf.reshape(current_input, [-1, conv_output_size])\
                            + b_conv, self.keep_prob)

        #create two fully-connected layers
        fc1_size=5
        W_fc1 = weight_variable([conv_output_size, fc1_size])
        b_fc1 = bias_variable([fc1_size])
        W_o = 10*weight_variable([fc1_size, 1])
        b_o= 10*bias_variable([1])
        fc1 = tf.nn.relu(tf.matmul(conv_output_flat, W_fc1) + b_fc1)
        self.y = tf.matmul(fc1, W_o) + b_o

        #squared-error loss function
        loss =  tf.reduce_mean(tf.square(self.y-self.y_))

        #define error as (y-y')/y'
        outy = (self.y*stddev)+mean
        outy_ =(self.y_*stddev)+mean
        self.error = abs(tf.reduce_mean(abs(outy-outy_))/tf.reduce_mean(outy_))

        self.train_step = tf.train.RMSPropOptimizer(\
            learning_rate = self.learning_rate,\
             momentum = 0.1, decay=0.5).minimize(loss)

        self.saver = tf.train.Saver()
        if not os.path.exists("saved"):
            os.makedirs("saved")
    
    def start_session(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        return self.sess
        
    def train_set(self, trainX, trainY):
        '''
        Performs training on the training batch given
        
        Returns training accuracy
        '''
        a, score = self.sess.run((self.train_step, self.error),\
            feed_dict={self.x: trainX, self.y_: trainY, self.keep_prob: 0.5})
        return score

    def test_set(self, testX, testY):
        '''
        Performs testing on the batch given
        
        Returns testing accuracy and predictions
        '''
        score = self.sess.run((self.error),\
            feed_dict={self.x: testX , self.y_: testY, self.keep_prob: 1.0 })
        return score
   
    def predict_set(self, predX, predY):
        return self.sess.run(self.y,\
            feed_dict={self.x: predX , self.y_: predY, self.keep_prob: 1.0 })
 
     
    def save_model(self):
        self.saver.save(self.sess, "saved/CNN.ckpt")
        
    def restore_model(self):
        self.saver.restore(self.sess, "saved/CNN.ckpt")
        
