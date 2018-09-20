import numpy as np
import pandas as pd

def Boxcar_Kernel(kernel_length=21.):
    return np.ones(int(kernel_length))/float(kernel_length)

def Evenly_Sample_Classes(X, y, max_class_samples=25):

    ix = np.arange(len(y))
    
    classes = np.unique(y)
    final_indices = np.array((),dtype=np.int64)
    for k in classes:
        np.random.shuffle(ix)
        sub_ix = np.where(y[ix]==k)[0][:max_class_samples]
        final_indices = np.concatenate([final_indices, ix[sub_ix]])
        
    np.random.shuffle(final_indices)
    
    return X[final_indices,:], y[final_indices]

def Exponential_Kernel(kernel_size=21, exp_factor=1.):
    #kernel_size = length of kernel
    #exp_factor controls the shape of the kernel, larger values bias coinvolution
    #product toward the end, i.e., more recent 
    kernelx = np.linspace(0.,1., kernel_size)
    kernel = np.exp(exp_factor*kernelx)
    return kernel / sum(kernel)

def get_one_minibatch(x_in, y_in, mini_batch_size):
    #get_one_minibatch: get one minibatch of data given the specified mini batch
    #size as well as provided X and Y in
    #get_one_minibatch(x,y), returns x and y minibatches
    ix = np.random.randint(0, np.shape(x_in)[0]-mini_batch_size)
    return x_in[ix:ix+mini_batch_size,:], y_in[ix:ix+mini_batch_size]

def get_YoY_Delta(dates):
    #get <timedelta> of all observations
    #use this function to find the shift needed to calculate a YoY change, this is just
    #a convenience function instead of having to manually enter the period needed to
    #calculate YoY
    dx = 0.0
    for k in range(1,len(dates)):
        dx = dx + (dates[k]-dates[k-1]).days
    #sampling rate (SR) = 365 days / <timedelta>
    SR = int ( round( 365.0 / ( dx / (len(dates)-1) ) ) )
    return SR

def Moving_Average(series, kernel):
    #compute a convolution (centered on the final value over each kernel) on
    #the input pandas dataseries, series. a new pd.Series will be returned
    #holding the new index and values convolved with kernel
    
    t = series.index
    Y = series.as_matrix()

    Y_conv = np.correlate(Y, kernel, 'full');
    Y_conv = Y_conv[len(kernel)-1:-len(kernel)+1]
    t_conv = t[len(kernel)-1:]
    
    return pd.Series(index=t_conv,data=Y_conv)
    
def shuffle_data(x_in, y_in):
    #shuffle_data: function used to shuffle X and y data
    #shuffle_data(x,y), returns shuffled x,y
    
    ix = np.arange(len(y_in))
    np.random.shuffle(ix)
    return x_in[ix,:], y_in[ix]

def train_test_split(X, y, n_sections=5, test_size=0.15, reverse_data=False):
    #create different training and test sets
    # boundaries = np.floor(np.linspace(0,np.shape(x)[0],6)).astype(int)

    if reverse_data:
        X = X[::-1,:]
        y = y[::-1]

    X_test = []
    y_test = []
    X_train = []
    y_train = []
    section_size = int(np.floor(np.shape(X)[0]*test_size/n_sections))

    test1 = 0
    rand_interval = int(np.shape(X)[0]/n_sections)

    for k in range(n_sections):

        train0 = k*rand_interval
        train1 = np.random.randint(train0, train0+rand_interval-section_size) 
        X_train.append( X[train0:train1,:] )
        y_train.append( y[train0:train1] )

        test0 = train1
        test1 = test0 + section_size
        X_test.append( X[test0:test1,:] )
        y_test.append( y[test0:test1] )
    
        #add any remaining data to the end
        X_train.append( X[test1:(k+1)*rand_interval,:] )
        y_train.append( y[test1:(k+1)*rand_interval] )

    if reverse_data:        
        X_train = [x[::-1,:] for x in X_train][::-1]
        y_train = [x[::-1] for x in y_train][::-1]
        X_test = [x[::-1,:] for x in X_test][::-1]
        y_test = [x[::-1] for x in y_test][::-1]

    X_train = np.vstack(X_train)
    y_train = np.ravel( np.concatenate(y_train) )

    X_test = np.vstack(X_test)
    y_test = np.ravel( np.concatenate(y_test))
    
    return X_train, y_train, X_test, y_test