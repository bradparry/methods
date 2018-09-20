import numpy as np
import tensorflow as tf
import data_processing as dp

from abc import ABCMeta
from abc import abstractmethod


class TF_Graph_Wrapper(object):
    """
    A wrapper for Tensorflow graphs that provides functionality for rapid saving 
    restoring, and fitting with early stopping from cross-validated metrics.
    
    Tensorflow is a powerful symbolic library, but is often unwieldly for smaller 
    projects due to graph construction and bottlenecks related to saving/restoring 
    checkpoints. This wrapper and child functions seeks to make Tensorflow graphs 
    more suitable for interactive ML projects by dramatically reducing build, 
    train and restore time.
    
    By default, training is performed with early-stopping determined on 
    internally calculated cross-validation sets. As training continues, optimal 
    graph values are updated and saved in memory as training loss decreases. 
    When training completes, the optimal graph values are loaded from memory back 
    into the graph (this takes a small fraction of the time it would normally 
    take to save Tensorflow checkpoints during training and reload the best at 
    the conclusion of training). The user may optionally create tensorflow 
    checkpoints.


    Available arguments/parameters: (default)
    _________________________________________
    activation: (tf.nn.elu) activation function for hidden layers
    
    alpha: (0.001) parameter to control the strength of regularization. May be 
    set to 0 to prevent parameter regularization
    
    batch_normalization: (False) bool specifiying wheter or not to perform 
    batch normalization (Ioffe & Szegedy, Batch Normalization: Accelerating 
    Deep Network Training by Reducing Internal Covariate Shift, 2015.)
    
    create_tensorboard_summaries: (np.inf) create tensorboard summaries every
    create_tensorboard_summaries iterations. setting create_tensorboard_summaries
    to np.inf blocks summary creation
    
    cross_validation_fraction: (0.1) the fraction of data to be used for test
    sets in cross validation
    
    early_stopping: (np.inf) step training prior to completion of n_steps of
    training if (current iteration - iteration of best loss) > early stopping.
    In other words, if training hasn't improved within the number of iterations 
    specified by early_stopping, training will end. If early_stopping > n_steps, 
    all n_steps will complete and there will be no early_stopping.
    
    layer_sizes: ([10,10]) a list of integers specifying the number of hidden 
    units in each hidden layer
    
    learning_rate: (0.01) learning rate for optimization algorithm
    
    loss_type: (L2) string specifying type of loss to minimize, must be either
    'L1' or 'L2'
    
    mini_batch_size: (250) the number of samples to be used on each training
    run
    
    n_cross_validation_sections: (5) the number of independent sections to
    form the test cross-validation set from
    
    n_input_dimensions: the number of input dimensions in the dataset. If fit()
    is called, this parameter is not necessary as the object will measure the
    shape of the input space
    
    n_steps: (1000) the maximum number of training iterations to perform
    
    pre_shuffle: (False) Shuffle the data prior to identifying cross-validation
    sets. In general, temporally correlated data should not be shuffled.
    
    shuffle: (False) whether or not the data should be shuffled on each
    training iteration. If True, shuffling occurs AFTER cross-validation sets
    have been determined
    
    use_cross_validation: (True) whether or not to perform cross-validation on
    the input data
    
    Available attributes
    ____________________
    
    NOTE: Several attributes are available to tweak behavior of the object. 
    However, the minimum necessary attributes for use are fit and predict.
    
    build_graph: construct the tensorflow graph used for learning and 
    inference. The size of the feature space must be provided prior to graph 
    construction. see n_input_dimensions above.
    
    fit: complete all training steps as specified by provided arguments and
    parameters. X must be a matrix with samples down the rows and features down
    the columns. restart_training arg specifies whether the graph should be re-
    initialized (True) or if training should continue from the last point (False).
    IF you run fit and receive a message that fitting has not converged, you
    can continue fitting with fit(X,y,restart_training=False).
    fit(x,y,restart_training), returns nothing
    
    fit_: perform one step of training and increment the internal training step
    count. used internally by fit()
    
    local_restore: restore the latest saved kernel and bias values from the
    internally saved dictionary (_tr_vars)
    
    local_save: extract learned kernels and biases and save them in an internal
    dictionary (_tr_vars) for later use. If you want to avoid tf checkpoints, 
    you could pickle the _tr_vars dictionary, build a new graph (build_graph()),
    unpickle the _tr_vars dictionary, sess.run(tf.global_variables_initializer())
    and be on your way for much faster than writing and restoring a tf
    checkpoint. Keep in mind this a high-level use of the graph and you miss
    the low-level details and re-usability of the tf checkpoint....
    local_save()
    
    predict: use the trained model to infer values from provided data
    predict(x), returns inferred values
    
    reload_tf_checkpoint: reload a tensorflow checkpoint given the path of a
    tensorflow checkpoint
    see https://www.tensorflow.org/guide/saved_model
    reload_tf_checkpoint(path_name), no return
    
    saved_step: the training iteration where the last kernel/weights snapshot
    was stored
    
    save_tf_checkpoint: save a tensorflow checkpoint at the specified path name
    save_tf_checkpoint(path_name) 
    see: https://www.tensorflow.org/guide/saved_model
    no return
    
    
    summary_dir: (.) the path to store training/test summaries. Summaries are
    created for both test and training batches (identical if no cross-validation
    is performed) and may be visualized by going to a shell and:
        >>> tensorboard --logdir=[summary_dir]
        Then open a browser window and navigate to http://localhost:6006
    
    
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, learning_rate=0.01, alpha=0.0001, layer_sizes=[100], n_input_dimensions=1, n_steps=10000, mini_batch_size=25, shuffle=False, early_stopping=200, regularization_type='L2', batch_normalization=False, create_tensorboard_summaries=np.inf, summary_dir=r'.'):
        self.activation = tf.nn.elu
        self.alpha = alpha
        self.batch_normalization = batch_normalization
        self.batch_norm_momentum = 0.9
        self.create_tensorboard_summaries = create_tensorboard_summaries
        self.cross_validation_fraction = 0.1
        self.early_stopping = early_stopping
        self.uniform_class_representation = None
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.n_cross_validation_sections = 5
        self.n_input_dimensions = n_input_dimensions
        self.n_steps = n_steps
        self.pre_shuffle = False
        self.shuffle = shuffle
        self.summary_dir = summary_dir
        self.use_cross_validation=True

        if regularization_type is not None:
            self.regularization_type = regularization_type.upper()
        else:
            self.regularization_type = None

        self.is_graph_available = False        
        self.training_step = 0
    
    @abstractmethod
    def build_graph(self):
        """
        construct a tensorflow graph that has inputs:
            x: X feature space
            y: y fit target
            is_training: bool if training or not
            
        has the following output:
            output
            loss
            predictions
            training_op
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess = tf.InteractiveSession()
            is_graph_available=True
        
        and maps all of these attributes to self without returning anything 
        explicitly
        
        """
        pass
    
    @abstractmethod
    def get_training_samples(self,x,y):
        """
        given inputs x,y which are data features and targets respectively,
        return some x,y to feed into the fit method.
        """
        pass
    
    def fit(self, x_in, y_in, restart_training=True):
        
        if self.pre_shuffle:
            x_in, y_in = dp.shuffle_data(x_in, y_in)
            
        if self.use_cross_validation:
            #If using cross_validation, create training and test groups
            X_train, y_train, X_test, y_test = dp.train_test_split(x_in, y_in, n_sections=self.n_cross_validation_sections, test_size=self.cross_validation_fraction)
            #make the test/train sets attributes in case the user wants to run stats on them
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
        else:
            #otherwise, make test and train groups identical
            X_train = X_test = x_in
            y_train = y_test = y_in

        if not self.is_graph_available:
            #used for classification tasks
            self.n_classes = len( np.unique(y_in) )
            self.n_input_dimensions = np.shape(x_in)[1]
            self.build_graph()

        if restart_training:
            self.sess.run(tf.global_variables_initializer())
            self.best_loss = np.inf
            self.L = []
            
        if self.create_tensorboard_summaries < np.inf:
            train_writer = tf.summary.FileWriter(self.summary_dir+r'/train', tf.get_default_graph())
            test_writer = tf.summary.FileWriter(self.summary_dir+r'/test', tf.get_default_graph())
        
        for k in range(1,self.n_steps+1):
                
            x0,y0 = self.get_training_samples(X_train, y_train) 

            self.fit_(x0,y0)
            
            self.L.append(self.sess.run(self.loss, feed_dict={self.x:X_test, self.y:y_test}))
            if self.L[-1] < self.best_loss:
                #if this is the best loss metric in training, save that value
                #and create an internal checkpoint
                self.best_loss = self.L[-1]
                self.local_save()
            
            if k % self.create_tensorboard_summaries == 0:
                summary = self.sess.run(self.merged, feed_dict={self.x:X_train, self.y:y_train, self.is_training:False})
                train_writer.add_summary(summary,k)
                
                summary = self.sess.run(self.merged, feed_dict={self.x:X_test, self.y:y_test, self.is_training:False})
                test_writer.add_summary(summary,k)
                
            if k - np.argmin(self.L) > self.early_stopping:
                break

        if k == self.n_steps:
            print('Training completed without convergence')
            
        #when training completes, reload the best graph snapshot
        self.local_restore()
    
    def local_restore(self):
        def_graph = tf.get_default_graph()

        feed_dict = {}
        for nm in list(self._tr_vars):
            feed_dict[def_graph.get_operation_by_name(nm.replace(':0','/Assign')).inputs[1]] = self._tr_vars[nm]

        init = tf.global_variables_initializer()
        self.sess.run(init, feed_dict=feed_dict)
        
    def local_save(self):
        #construct a dictionary to save all kernels and biases from the 
        #tensorflow graph
        self._tr_vars = {}
        for trainable_var in tf.get_default_graph().get_collection('trainable_variables'):
            self._tr_vars[trainable_var.name] = trainable_var.eval()
        
        self.saved_step = self.training_step
    
    def save_tf_checkpoint(self, path_and_name):
        self.saved_path = self.saver.save(self.sess, path_and_name)
        print('checkpoint saved at:\n\t'+self.saved_path)
        
    def reload_tf_checkpoint(self, path_and_name):
        tf.reset_default_graph()
        self.build_graph()
        self.saver.restore(self.sess, path_and_name)
    
    def fit_(self, x_in, y_in):
        self.sess.run([self.training_op, self.update_ops], feed_dict={self.is_training:True, self.x:x_in, self.y:y_in})
        self.training_step += 1
            
    def predict(self, x_in):
        return np.ravel(self.sess.run(self.predictions, feed_dict={self.x:x_in, self.is_training:False}))      

        

class NN_Classifier(TF_Graph_Wrapper):
    
    """
    Construct a neural network classifier in Tensorflow with a user-specified
    graph, train the graph and infer classification. By default, the model is 
    cross-validated and the test sets are used for early stopping.
    
    Trained values are saved internally to accelerate training/inference time
    in tensorflow. The internal saving procedure takes a small fraction of the
    time that it takes to write a full tensorflow checkpoint. The idea was to 
    make tensorflow graph optimization and checkpoint saving faster such that 
    training and inference could be more interactive in tensorflow. The user 
    may optionally create tensorflow checkpoints.
    
    At the end of training (i.e., after NN_Classifier.fit() completes), the 
    best graph parameters are automatically reloaded into the graph so that the
    user may immediately use the graph for inference (i.e., 
    NN_Classifier.predict())). See below for an example.
    
    See attributes and methods from TF_Graph_Wrapper
    
    Minimal Example
    _______________
    import numpy as np

    #create artificial data with 2 features
    n_classes = 5
    samples_per_class = 300
    X = []
    y = [np.array((),dtype=np.int64)]
    for k in range(n_classes):
        X.append(np.hstack((np.random.normal(3*k,1,size=[samples_per_class,1]),np.random.normal(2.5*np.sin(k*1.5),1,size=[samples_per_class,1]))))
        y.append(np.zeros(samples_per_class)+k)
    X = np.vstack(X)
    y = np.concatenate(y)
    
    nn = NN_Classifier()
    nn.fit(X,y)
    #get predictions
    yhat = nn.predict(X)
    #get accuracy on the entire dataset
    accuracy = nn.sess.run(nn.percent_correct,feed_dict={nn.x:X, nn.y:y})
    #get accuracy on the training set
    accuracy_train = nn.sess.run(nn.percent_correct,feed_dict={nn.x:nn.X_train, nn.y:nn.y_train})
    #get accuracy on the test set
    accuracy_test = nn.sess.run(nn.percent_correct,feed_dict={nn.x:nn.X_test, nn.y:nn.y_test})
    """
    
    def __init__(self, **kwargs):
        super(NN_Classifier, self).__init__(**kwargs)
    
    def build_graph(self):
        tf.reset_default_graph()
        
        with tf.name_scope('inputs'):
            x = tf.placeholder(dtype=tf.float32,shape=[None,self.n_input_dimensions],name='x')
            y = tf.placeholder(dtype=tf.int64,shape=[None],name='y')
            is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        
        with tf.name_scope('hidden_layers'):
            #get he initializer
            initializer = tf.variance_scaling_initializer()
            
            #add all hidden layers to a list
            hidden_layers = [x]
            if self.batch_normalization:
                #construct hidden layers with batch normalization
                for layer_number,layer_sz in enumerate(self.layer_sizes):
                    tmp = tf.layers.dense(hidden_layers[-1], layer_sz, kernel_initializer=initializer, name='layer'+str(layer_number))
                    bn = tf.layers.batch_normalization(tmp, training=is_training, momentum = self.batch_norm_momentum)
                    hidden_layers.append( self.activation(bn) )
                #add the final layer which should have no activation function and be
                #one dimension to transform the output to a singular value
                tmp = tf.layers.dense(hidden_layers[-1], self.n_classes, kernel_initializer=initializer)
                output = tf.layers.batch_normalization(tmp, training=is_training, momentum=self.batch_norm_momentum)
                
            else:
                for layer_number,layer_sz in enumerate(self.layer_sizes):
                    #construct hiddenf layers ...
                    hidden_layers.append( tf.layers.dense(hidden_layers[-1], layer_sz, kernel_initializer=initializer, activation=self.activation, name='layer'+str(layer_number)) )
                output = tf.layers.dense(hidden_layers[-1], self.n_classes, kernel_initializer=initializer)
            
        with tf.name_scope('evaluate_model'):
            value, predictions = tf.nn.top_k(output, 1, name='predictions')
            correct = tf.nn.in_top_k(output, y,1,name='correct')
            percent_correct = tf.reduce_mean(tf.cast(correct, tf.float32), name='percent_correct')
            tf.summary.scalar('percent_correct',percent_correct)
        
        with tf.name_scope('loss'):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output, name='xentropy')
            mean_xentropy = tf.reduce_mean(xentropy, name='mean_xentropy')
            
            #grab kernel values for regularization. do not use built-in 
            #tensorflow regularization to avoid interfering with batch 
            #normalization. Explicitly, only grab data from hidden layer kernels
            kernels = []
            for tr_var in tf.trainable_variables():
                if tr_var.name.find('kernel') > -1:
                    kernels.append(tf.reshape(tr_var,[-1]))
                    #add a histogram summary for each layer
                    tf.summary.histogram(tr_var.name.replace(':','_'), tf.reshape(tr_var,[-1]))

            #reshape kernel values for regularization
            kernel_values = tf.concat(kernels, -1, name='kernel_values')
            
            #perform specified regularization
            reg_penalty = 0.
            if self.regularization_type == 'L1':
#                print('L1 Regularization')
                reg_penalty = tf.multiply( self.alpha, tf.reduce_sum( tf.abs(kernel_values) ), name='reg_penalty')
            elif self.regularization_type == 'L2':
#                print('L2 Regularization')
                reg_penalty = tf.multiply( self.alpha, tf.reduce_sum( tf.square(kernel_values) ), name='reg_penalty')
            else:
                pass
#                print('No parameter regularization')

            #construct the loss by adding prediction differences and regularization penalty
            loss = tf.add( mean_xentropy, reg_penalty, name='loss' )
            
        with tf.name_scope('training'):    
            adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam_optimizer')
            training_op = adam_optimizer.minimize(loss,name='training_op')            
        
        #save a bunch of graph ops into the object
        self.merged = tf.summary.merge_all()
        self.percent_correct = percent_correct
        self.x = x
        self.y = y
        self.is_training = is_training
        self.output = output
        self.loss = loss
        self.predictions = predictions
        self.training_op = training_op
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.init = tf.global_variables_initializer()

        self.sess = tf.InteractiveSession()
        self.is_graph_available = True
        
    def get_training_samples(self,x,y):
        class_samples = int(self.mini_batch_size/float(self.n_classes))
        return dp.Evenly_Sample_Classes(x, y, max_class_samples=class_samples)


class NN_Regressor(TF_Graph_Wrapper):
    
    """
    Construct a tensorflow graph to perform N-layer perceptron regression, 
    with user-specified loss, regularization, activation, with tensorboard
    summaries, etc, etc... By default, the model is cross-validated and the 
    test sets are used for early stopping.
    
    Trained values are saved internally to accelerate training/inference time
    in tensorflow. The internal saving procedure takes a small fraction of the
    time that it takes to write a full tensorflow checkpoint. The idea was to 
    make tensorflow graph optimization and checkpoint saving faster such that 
    training and inference could be more interactive in tensorflow. The user 
    may optionally create tensorflow checkpoints.
    
    See attributes and methods from TF_Graph_Wrapper
    
    Minimal Example
    _______
    import numpy as np
    import matplotlib.pyplot as plt
    import NN_regression as NN
    
    X = np.linspace(0,1.8*np.pi,2000)
    y = np.sin(X) +np.exp(X/(3))+ np.random.normal(scale=0.05,size=len(X))
    X = (X-np.mean(X)) / np.std(X)
    y = (y-np.mean(y)) / np.std(y)
    X = np.vstack(X)
    
    nn = NN.NN_Regressor(shuffle=True)
    nn.fit(X,y)
    yh = nn.predict(X)
    
    plt.plot(X,y,label='raw_data')
    plt.plot(X,yh,'r',label='fit')
    plt.legend()
    
    
    Available arguments/parameters beyond the parent: (default)
    _________________________________________
    
    loss_type: (L2) string specifying type of loss to minimize, must be either
    'L1' or 'L2'
    """
    
    def __init__(self, loss_type='L2', **kwargs):
        super(NN_Regressor, self).__init__(**kwargs)
        self.loss_type=loss_type
        
    def build_graph(self):
        tf.reset_default_graph()
        
        with tf.name_scope('inputs'):
            x = tf.placeholder(dtype=tf.float32,shape=[None,self.n_input_dimensions],name='x')
            y = tf.placeholder(dtype=tf.float32,shape=[None],name='y')
            is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        
        with tf.name_scope('hidden_layers'):
            #get he initializer
            initializer = tf.variance_scaling_initializer()
            
            #add all hidden layers to a list
            hidden_layers = [x]
            if self.batch_normalization:
                #construct hidden layers with batch normalization
                for layer_number,layer_sz in enumerate(self.layer_sizes):
                    tmp = tf.layers.dense(hidden_layers[-1], layer_sz, kernel_initializer=initializer, name='layer'+str(layer_number))
                    bn = tf.layers.batch_normalization(tmp, training=is_training, momentum = self.batch_norm_momentum)
                    hidden_layers.append( self.activation(bn) )
                #add the final layer which should have no activation function and be
                #one dimension to transform the output to a singular value
                tmp = tf.layers.dense(hidden_layers[-1], 1, kernel_initializer=initializer)
                predictions = tf.layers.batch_normalization(tmp, training=is_training, momentum=self.batch_norm_momentum)
                
            else:
                for layer_number,layer_sz in enumerate(self.layer_sizes):
                    #construct hiddenf layers ...
                    hidden_layers.append( tf.layers.dense(hidden_layers[-1], layer_sz, kernel_initializer=initializer, activation=self.activation, name='layer'+str(layer_number)) )
                predictions = tf.layers.dense(hidden_layers[-1], 1, kernel_initializer=initializer)
            
        with tf.name_scope('loss'):
            #calculate the loss on predictions (output) vs y. measure the loss
            #as either L1 or L2 depending on user specified argument
            if self.loss_type == 'L1':
#                print('L1 loss')
                prediction_differences = tf.abs( tf.reshape(predictions, [-1]) - y, name='prediction_differences')
            elif self.loss_type == 'L2':
#                print('L2 loss')
                prediction_differences = tf.square( tf.reshape(predictions, [-1]) - y, name='prediction_differences')
 
            elif self.loss_type == 'L1_2':
#                print('sqrt loss')
                prediction_differences = tf.sqrt( tf.abs( tf.reshape(predictions, [-1]) - y, name='prediction_differences') )
              
            else:
                #misunderstanding the loss arg should be a fatal error
                print('Loss type was not recognized. No graph was constructed.')
                tf.reset_default_graph()
                return
            
            #measure the mean difference between predicted and ground truth.
            #construct a summary scalar for tensorboard
            mean_differences = tf.reduce_mean(prediction_differences, name='mean_differences')
            tf.summary.scalar('mean_differences',mean_differences)
            
            #grab kernel values for regularization. do not use built-in 
            #tensorflow regularization to avoid interfering with batch 
            #normalization. Explicitly, only grab data from hidden layer kernels
            kernels = []
            for tr_var in tf.trainable_variables():
                if tr_var.name.find('kernel') > -1:
                    kernels.append(tf.reshape(tr_var,[-1]))
                    #add a histogram summary for each layer
                    tf.summary.histogram(tr_var.name.replace(':','_'), tf.reshape(tr_var,[-1]))

            #reshape kernel values for regularization
            kernel_values = tf.concat(kernels, -1, name='kernel_values')
            
            #perform specified regularization
            reg_penalty = 0.
            if self.regularization_type == 'L1':
#                print('L1 Regularization')
                reg_penalty = tf.multiply( self.alpha, tf.reduce_sum( tf.abs(kernel_values) ), name='reg_penalty')
            elif self.regularization_type == 'L2':
#                print('L2 Regularization')
                reg_penalty = tf.multiply( self.alpha, tf.reduce_sum( tf.square(kernel_values) ), name='reg_penalty')
            else:
                pass
#                print('No parameter regularization')

            #construct the loss by adding prediction differences and regularization penalty
            loss = tf.add( tf.reduce_sum(prediction_differences), reg_penalty, name='loss' )
            
        with tf.name_scope('training'):    
            adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam_optimizer')
            training_op = adam_optimizer.minimize(loss,name='training_op')            
        
        #save a bunch of graph ops into the object
        self.merged = tf.summary.merge_all()
        self.x = x
        self.y = y
        self.is_training = is_training
        self.predictions = predictions
        self.loss = loss
        self.prediction_differences = prediction_differences
        self.mean_differences = mean_differences
        self.training_op = training_op
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.init = tf.global_variables_initializer()

        self.sess = tf.InteractiveSession()
        self.is_graph_available = True
    
    def get_training_samples(self,x,y):
        if self.shuffle:
            x, y = dp.shuffle_data(x, y)
            
        return dp.get_one_minibatch(x, y, self.mini_batch_size)
        