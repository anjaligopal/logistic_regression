import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import io
import time

def feature_append(data):
    '''
    Appends a column vector of 1s to the feature vector.
    '''

    # This function 
    data = np.append(data,np.ones((data.shape[0],1)),1)
    return(data);

def standardize_features(training_data,test_data):
    '''
    This function normalizes features to be of 0 mean
    and unit variance
    '''
    
    data_mean = np.average(training_data,axis=0);
    data_std = np.std(training_data,axis=0);
    training_data = (training_data - data_mean)/data_std;
    test_data = (test_data - data_mean)/data_std;
    return(training_data, test_data);


def data_shuffle(X,y):
    '''
    Shuffles data (X) and labels (y)
    Assumes shuffling dimension is dim 0
    '''

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X.copy()[indices,:]
    y_shuffled = y.copy()[indices,:]
    return(X_shuffled,y_shuffled)


def validation_partition(data,labels,n):

    '''
    Splits a data-frame with the first n values used for the validation data set.
    '''

    n = int(n)

    training_data = data[0:n]
    training_labels = labels[0:n]
    validation_data = data[n:]
    validation_labels = labels[n:]
    return(training_data, training_labels, validation_data, validation_labels);

def error_rate(predicted_class,true_class):
    # funciton to calculate the error rate
    error = (len(true_class) - np.sum(predicted_class==true_class))/len(true_class);
    return(error)

def accuracy(predicted_class,true_class):
    ''' 
    Calculates accuracy 
    ''' 
    accuracy = (np.sum(predicted_class==true_class))/len(true_class);
    return(accuracy)

def expit(x):
    return(1/(1+np.exp(-x)));

def sigmoid(x):
    y = 1/(1+np.exp(-x));
    return(y)

class LogisticRegression():
    '''
    Performs logistic regression.
    
    Inputs: 
    
    '''
    
    def __init__(self):
        return None       
    
    def train(self,X,y,lr,reg,iterations,batch_size=None):
        '''
        Trains the model.

        Inputs:
        - X: training data
        - y: classes
        - lr: learning rate (epsilon)
        - reg: regularization 
        - iterations: number of iterations
        - batch size: if set to none, uses whole batch

        Returns:
        - Weights
        - Predicted outputs 
        - Cost array
        '''
        
        import numpy as np
        
        if batch_size == None:
            batch_size = X.shape[0];
        elif batch_size > X.shape[0]:
            raise LookupError("Batch size greater than number of samples")
            
        # Getting the number of features (dimensions)
        n = X.shape[0]
        dim = X.shape[1]

        #  Formatting data
        X = np.matrix(X);
        y = np.matrix(y);
         
        # Initializing variables
        w = np.matrix(np.zeros((dim,1)))
        i = 0;
        cost_array = []
    
        # Getting information on batches       
        batches_per_epoch = np.ceil(n/batch_size); 
        print("batches per epoch: ",batches_per_epoch)
        batch_indices = [0, batch_size];
 
        start = time.time()
 
        while i < iterations:
            
            # Shuffle data after every epoch
            if (i/batches_per_epoch %1 == 0): 
                batch_indices = [0, batch_size];
                X, y = data_shuffle(X,y)
            
            # Getting the batches

            if batch_size == n:
                # If not doing SGD, batches are the whole
                # sample set
                X_batch = X
                y_batch = y
            
            elif (i % batches_per_epoch == batches_per_epoch - 1): 
                # If it's the last mini-batch in the epoch, get all 
                # remaining sample points. Otherwise, grab points
                # via the indices in batch_indices. 

                X_batch = X[batch_indices[0]:-1]
                y_batch = y[batch_indices[0]:-1]
                
            else:
                X_batch = X[batch_indices[0]:batch_indices[1]]
                y_batch = y[batch_indices[0]:batch_indices[1]];
                
            local_batch_size = X_batch.shape[0]

            
            # Calculating updates on weights
            X_times_w = X_batch*w;
            s = expit(X_batch*w);
            w = w - lr*(2*reg*w - X_batch.T*(y_batch-s));

            # Calculating the Cost

            ## note: if (1-s) = 0, then I set log(1-s) = (X_batch*w)_i, where
            ## i corresponds to the index where 1-s = 0; 
            # To do this, we first set 1-s = 1, where taking the log of that is zero 
            X_times_w = X*w;
            s = expit(X*w);
            
            one_minus_s = 1.0 -s;

            # Finding where 1 - s = 0
            log_zero_positions = one_minus_s == 0;

            # Converting the values of these positions to 1
            one_minus_s[log_zero_positions] = 1.0;

            # Taking the log of the new array
            log_one_minus_s = np.log(one_minus_s).reshape(n,1);

            # Using matrix multiplication to convert the 1 - s = 0 positionns to
            log_zero_values = np.multiply(np.array(log_zero_positions).reshape(n,1), X_times_w).reshape(-1,1)


            # Adding these to the log(1-s) array
            log_one_minus_s = log_one_minus_s + log_zero_values

            # Calculating cost 
            cost = float(lr*w.T*w - y.T*np.log(s) - (1-y).T * log_one_minus_s);

            cost_array.append(cost);

            # Updating the loop variables 
            batch_indices = [batch_indices[1], batch_indices[1]+batch_size];
            i += 1;
            
        self.weights = w;
        print("Time to complete: ",time.time()-start)
        y_pred_training = np.round(sigmoid(np.dot(X,self.weights)));
        return([self.weights,y_pred_training,cost_array]);

    def calculate(self,X):
        '''
        Calculates classes based on trained model
        '''

        if hasattr(self,'weights') == False:
            raise AttributeError("Please train the model first.")

        y = np.round(sigmoid(np.dot(X,self.weights)));
        return(y)

