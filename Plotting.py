"""
Unused functions that might be useful some day
"""

def plot_predictions():
    '''
    Plots predictions and actual data for 5 files randomly chosen
     from the test set
    '''
    for file in test[-5:]:
      df=np.loadtxt(file, delimiter=',')
      Y = df[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      X = df[:, :-3]
      Y = (Y-mean)/float(stddev)
      nbatches = int(len(Y)/batch_size)
      predict=np.array([]).reshape(0,1)
      for i in range(nbatches):
            batchX = X[i*batch_size:(i+1)*batch_size,:]
            batchY = Y[i*batch_size:(i+1)*batch_size]
            predict = np.concatenate((predict,predict_set(batchX,batchY)))
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
      df=np.loadtxt(file, delimiter=',')
      Y = df[:, y_col]
      Y = np.reshape(Y, (len(Y),1))
      X = df[:, :-3]
      
      #Normalize y
      Y = (Y-mean)/float(stddev)
      predict=np.array([]).reshape(0,1)
      nbatches = int(len(Y)/batch_size)
      for i in range(nbatches):
            batchX = X[i*batch_size:(i+1)*batch_size,:]
            batchY = Y[i*batch_size:(i+1)*batch_size]
            predict = np.concatenate((predict,predict_set(batchX,batchY)))
      plt.scatter(Y,predict, c='k', marker='.', s=1)
    plt.plot([-5,5],[-5,5],'r')
    plt.show()
