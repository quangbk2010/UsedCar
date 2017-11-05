#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:41:56 2017

@author: quang
"""
import tensorflow as tf
import argparse
import os
from Header_lib import *
from Used_car_regression import *

class Evaluation ():
    
    def __init__ (self):
        pass
    
    def get_regression_predictions(self, regr, Xbar):
        """
            input_feature: an array contains name of features (coresponds to a matrix, with each row is a data point)
            w: weights
            => return: predicted output
        """    
        predicted_output = regr.predict (Xbar) 
        return predicted_output

    def get_residual_sum_of_squares(self, predicted_output, y):
        """
        Calculate RSS when know input and label of test data, awa weights
        """
        RSS = sum ((y - predicted_output) ** 2)
        return RSS[0]
       
    def get_RMSE(self, predicted_output, y):
        """
            Calculate Square Root of MSE (Mean of RSS) when know input and label of test data, awa weights
        """
        return np.sqrt (mean_squared_error(predicted_output, y))
        
    def get_MAE(self, predicted_output, y):
        """
            Calculate Mean Absolute Error when know input and label of test data, awa weights
        """
        return mean_absolute_error(predicted_output, y)
      
    
    def get_score_of_model (self, regr, X_test, y):
        """
            X_test: the expanded data matrix
            y: vector of outputs
        """
        return regr.score(X_test,y)
    
    
    def get_pearsonr_correlation (self, array1, array2):
        return pearsonr (array1, array2)
    
    
    def get_anova_correlation (self, data):
        
        # Create a boxplot
        plt.figure()
        fig, ax = plt.subplots(figsize=(10,  5))
        
        data.boxplot('price', by='manufacture_code', ax=ax)       
        
        plt.savefig('C:\\Users\\user\\Google Drive\\Code\\Projects\\UsedCar\\Results\\plot.png')
        
    
    def is_significant_improved (self, x0, x, significant_value):
        """
            - Purpose:
            - Input:
                + x0: initial value of x
                + x: the needed evaluated value of x
                + significant: a threshold (from (0-1)) to indicate how much x is improved from initial value
            - return: x is improved significantty or not.
        """
        if (1 - x / x0 >= significant_value):
            return 1
        return 0
    
class Tensor_NN(Dataset):

    def __init__(self, args):
        #from args
        self.gpu_idx = args.gpu_idx

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.k_fold = args.k_fold
        self.dropout = args.dropout


        self.learning_rate = args.learning_rate
        self.decay_step = args.decay_step
        self.decay_rate = args.decay_rate
        self.saved_period = args.saved_period

        self.dim_data = args.dim_data
        self.dim_label = args.dim_label
        self.no_hidden_layer = args.no_hidden_layer
        self.no_neuron = args.no_neuron
        
        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.args = args

        Dataset.__init__(self)
        self.features = features

        
        self.feature_constraint = feature_constraint
        self.feature_constraint_values = feature_constraint_values

        if constraint_flag == 0:
            (self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set) = self.get_data_label (self.features, output)
        else:
            (self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set) = self.get_data_label_with_constraints (self.features, output, self.feature_constraint, self.feature_constraint_values)

    def get_data_label (self, features, output):
        """
            - Purpose: Devide total dataset into n_splits set.
                Using (n_splits - 1) sets for training, the remaining for testing
            - Input:
             + features: an array of feature names
             + output: name of the output (label)
            - Return: a tuple with each element contains a list of "n_splits" training sets (X_train_set, y_train_set) 
             and "n_splits" test sets (X_test_set, y_test_set)
        """
        
        
        # Due to the notes -> need to comment these 2 lines
        X_total_set = self.get_data_matrix (self.total_dataset, features) 

        scaler = StandardScaler()  
        scaler.fit(X_total_set)  
        X_total_set = scaler.transform(X_total_set)  
            

        if output == "price":
            y_total_set = self.get_data_array (self.total_dataset, output)
        elif output == "sale_duration":
            y_total_set = self.get_sale_duration_array (self.total_dataset)
        #print ("test:", self.total_dataset.shape, X_total_set.shape, y_total_set.shape)
        #sys.exit (-1)
        X_train_set = []
        y_train_set = []
        X_test_set = []
        y_test_set = []
        
        if self.k_fold > 0:
            kf = KFold(self.k_fold, shuffle=True)
            
            for train_index, test_index in kf.split(X_total_set):
                
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X_total_set[train_index], X_total_set[test_index]
                y_train, y_test = y_total_set[train_index], y_total_set[test_index]
                #print ("X_train shape:", X_train.shape, "y_train type:", y_train.shape)
                #print ("X_train:", X_train, "y_train:", y_train)
               
                X_train_set.append (X_train)
                y_train_set.append (y_train)
                X_test_set.append (X_test)
                y_test_set.append (y_test)

        else:
            X_train_set = X_total_set[:data_training_length, :]
            y_train_set = X_total_set[:data_training_length, :]

            X_test_set = X_total_set[data_training_length:, :]
            y_test_set = X_total_set[data_training_length:, :]

            
        return (X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set) 
    
    def get_data_label_with_constraints (self, features, output, feature_constraint, feature_constraint_values):
        """
            - Purpose: Devide total dataset into n_splits set.
                Using (n_splits - 1) sets for training, the remaining for testing
            - Input:
             + features: an array of feature names
             + output: name of the output (label)
             + feature_constraint: name of the feature to limit the data, label
             + feature_constraint_values: a list of accepted values
            - Return: a tuple with each element contains a list of "n_splits" training sets (X_train_set, y_train_set) 
             and "n_splits" test sets (X_test_set, y_test_set)
        """
        
        
        X_total_set = self.get_data_matrix_with_constraints (self.total_dataset, features, feature_constraint, feature_constraint_values)
        y_total_set = self.get_data_array_with_constraints (self.total_dataset, output, feature_constraint, feature_constraint_values)
        
        scaler = StandardScaler()  
        scaler.fit(X_total_set)  
        X_total_set = scaler.transform(X_total_set)  
           
        X_train_set = []
        y_train_set = []
        X_test_set = []
        y_test_set = []
        
        if self.k_fold > 0:
            kf = KFold(self.k_fold, shuffle=True)
            
            for train_index, test_index in kf.split(X_total_set):
                
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X_total_set[train_index], X_total_set[test_index]
                y_train, y_test = y_total_set[train_index], y_total_set[test_index]
                #print ("X_train shape:", X_train.shape, "y_train type:", y_train.shape)
                #print ("X_train:", X_train, "y_train:", y_train)
                X_train_set.append (X_train)
                y_train_set.append (y_train)
                X_test_set.append (X_test)
                y_test_set.append (y_test)
                
        else:
            X_train_set = X_total_set[:data_training_length, :]
            y_train_set = X_total_set[:data_training_length, :]

            X_test_set = X_total_set[data_training_length:, :]
            y_test_set = X_total_set[data_training_length:, :]
            
        return (X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set) 



    def build_model (self, dim_data, dim_label, no_unit_in_a_hidden_layer, no_hidden_layer):#dim_label = 1x
        weights = []

        X = tf.placeholder(tf.float32, [None, dim_data])
        Y = tf.placeholder(tf.float32, [None, dim_label])

        dropout = tf.placeholder(tf.float32, name='dropout')

        # Input layer
        W1 = tf.Variable(tf.random_normal([dim_data, no_unit_in_a_hidden_layer], stddev=0.01))
        B1 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer], stddev=0.01))
        L1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, W1), B1))

        weights.append (W1)
        weights.append (B1)

        # Hidden layers
        L2_hat = L1
        for i in range (no_hidden_layer):
            W2 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer, no_unit_in_a_hidden_layer], stddev=0.01))
            B2 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer], stddev=0.01))
            L2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(L2_hat, W2), B2))
            L2_hat = tf.nn.dropout(L2, dropout, name='relu_dropout')
            weights.append (W2)
            weights.append (B2)

        # Output layer
        W3 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer, dim_label], stddev=0.01))
        B3 = tf.Variable(tf.random_normal([dim_label], stddev=0.01))
        prediction = tf.nn.bias_add(tf.matmul(L2_hat, W3), B3)

        weights.append(W3)
        weights.append(B3)

        return X, Y, prediction, weights, dropout



    def train_nn(self, train_data, train_label, test_data, test_label, no_neuron, no_hidden_layer, dropout_val, model_path): #k_fold, 

       
        #building model
        X, Y, prediction, weights, dropout = self.build_model(train_data.shape[1], self.dim_label, no_neuron, no_hidden_layer)

        #evaluate model on training process
        """cost = tf.reduce_mean(tf.squared_difference(prediction, Y)) # or can use tf.nn.l2_loss
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost) #TODO: learning rate adaptive
        rmse = tf.sqrt(cost)

        """
        # Try to use weights decay
        num_batches_per_epoch = int(len(train_data) / self.batch_size)
        decay_steps = int(num_batches_per_epoch * self.decay_step)
        global_step = tf.Variable(0, trainable = False)
        #learning_rate = self.learning_rate
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, decay_steps, self.decay_rate, staircase = True)

        # Used for minimizing RMSE
        #loss = tf.reduce_mean(tf.squared_difference(prediction, Y))

        # Used for minimizing relative error
        loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), (Y))) #protect when Y = 0

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
        # RMSE
        rmse = tf.sqrt(loss)
        relative_err = loss * 100
        
        # Calculate root mean squared error as additional eval metric

        """ Initialize the variables with default values"""
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            saver = tf.train.Saver()

            # Run the initializer
            sess.run(init)

            """ Start training, using training data"""
            # record the time when training starts
            start_time = time.time()

            train_set = np.concatenate ((train_data, train_label), axis = 1)
            train_set_shuffled = np.random.permutation(train_set)
            train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
            train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

            total_batch = int((len(train_data)/self.batch_size) + 0.5)

            pre_epoch_test_relative_err_val = 0

            for epoch in range (self.epoch):

                total_relative_err = 0
                index_counter = 0
                left_num = len(train_data)
                for i in range (total_batch):
                    start_index = index_counter * self.batch_size
                    end_idex = 0
                    if(left_num < self.batch_size):
                        end_index = index_counter * self.batch_size + left_num
                    else:
                        end_index = (index_counter + 1) * self.batch_size

                    batch_x = train_data_shuffled [start_index : end_index]
                    batch_y = train_label_shuffled [start_index : end_index]
                    #print ("batch_x", batch_x)
                    #print ("batch_y", batch_y)
                    index_counter = index_counter + 1
                    left_num = left_num - self.batch_size

                    if (left_num <= 0):
                        index_counter = 0

                    #_, cost_val, lr = sess.run([optimizer, cost, learning_rate], feed_dict={X: batch_x, Y: batch_y})
                    _, training_relative_err_val = sess.run([optimizer, relative_err], feed_dict={X: batch_x, Y: batch_y, dropout:dropout_val})
                    total_relative_err += training_relative_err_val
                    
                    a, b = sess.run([Y, prediction], feed_dict={X: batch_x, Y: batch_y, dropout:dropout_val})
                    #print ("err:", a[0][0], b[0][0], training_relative_err_val)

                #print('Epoch: %04d' % (epoch + 1), 'Avg. rmse = {:.3f}'.format(total_rmse / total_batch), 'learning_rate = {:.5f}'.format(lr))
                print('Epoch: %04d' % (epoch + 1), 'Avg. training relative_err = {:.3f}%'.format(total_relative_err / total_batch))
                
                epoch_test_relative_err_val = sess.run(relative_err, feed_dict={X: test_data, Y: test_label, dropout:self.dropout})
                
                print (pre_epoch_test_relative_err_val, epoch_test_relative_err_val)
                #if epoch_test_relative_err_val > pre_epoch_test_relative_err_val and pre_epoch_test_relative_err_val != 0:
                    #break
                #pre_epoch_test_relative_err_val = epoch_test_relative_err_val

                #TODO: training data permutation
                train_set_shuffled = np.random.permutation(train_set)
                train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
                train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

                if (epoch + 1) % self.saved_period == 0 and epoch != 0:
                    #model_path = self.model_dir + '/' + self.model_name + '_' + str (k_fold) + '_' + str(epoch + 1) + '.ckpt'
                    save_path = saver.save(sess, model_path, global_step=global_step)
                    print('Model saved in file: %s' % save_path)

            print ("Training finished!")
            stop_time = time.time()
            print ("Training time (s):", stop_time - start_time)
            
            #test_relative_err_val = sess.run(relative_err, feed_dict={X: test_data, Y: test_label, dropout:self.dropout})
            print('test rmse: {:.3f}'.format(epoch_test_relative_err_val))
            return epoch_test_relative_err_val
   
    
    
    
    def test_CV (self, no_neuron, no_hidden_layer, dropout_val, model_path):
        
        with tf.Session() as sess:
            """ In each iteration, we need to store training error and test error into 2 separare lists"""
            test_err = []

            for i in range (self.k_fold):
                print ("fold:", i)
                train_data = self.X_train_set[i] 
                train_label = self.y_train_set[i]
                test_data = self.X_test_set[i]
                test_label = self.y_test_set[i]

                #print (train_data.shape, train_label.shape, test_data.shape, test_label.shape)
                #sys.exit (-1)
                #print (train_data)
                #print (train_label)
                #print (test_data)
                #print (test_label)
                #sys.exit(-1)
                
                train_set = np.concatenate ((train_data, train_label), axis = 1)
                train_set_shuffled = np.random.permutation(train_set)
                train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
                train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

                total_batch = int(train_data.shape[0]/self.batch_size)
                test_relative_err_val = self.train_nn(train_data_shuffled, train_label_shuffled, test_data, test_label, no_neuron, no_hidden_layer, dropout_val, model_path)
                test_err.append (test_relative_err_val) 
                #print ("---", test_relative_err_val)
        print ("---Mean relative_err:", np.mean (test_err))
        return np.mean (test_err)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #configuration
    parser.add_argument('--gpu_idx', type=str, default = '0')
    parser.add_argument('--model_dir', type=str, default = '../checkpoint')
    parser.add_argument('--model_name', type=str, default = 'usedCar')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--saved_period', type=int, default=100, help='save checkpoint per X epoch') # 200
    parser.add_argument('--output_dir', type=str, default = '../Results')

    #hyper parameter
    parser.add_argument('--epoch', type=int, default = 100) #2000
    parser.add_argument('--dropout', type=int, default = 1)
    parser.add_argument('--batch_size', type=int, default = 128) # 128
    parser.add_argument('--learning_rate', type=float, default=0.00125)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--decay_step', type=int, default=100) #if decay_step > epoch, no exponential decay

    #network parameter
    parser.add_argument('--dim_data', type=int, default=24)
    parser.add_argument('--dim_label', type=int, default=1)
    parser.add_argument('--no_hidden_layer', type=int, default = 1) #not implement variabel network layer
    parser.add_argument('--no_neuron', type=int, default = 1000)
    parser.add_argument('--k_fold', type=int, default = -1)

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx

    # record the time when training starts
    start_time = time.time()

    
    #print (data_training_length)
    #sys.exit (-1)
    nn = Tensor_NN (args)
    model_path = nn.model_dir + '/' + nn.model_name

    print ("Shape:", nn.X_total_set.shape, nn.y_total_set.shape)

    total_set = np.concatenate ((nn.X_total_set, nn.y_total_set), axis = 1)
    total_set_shuffled = np.random.permutation(total_set)
    total_data_shuffled = total_set_shuffled [:, 0:total_set.shape[1]-1]
    total_label_shuffled = total_set_shuffled [:, total_set.shape[1]-1:]

    train_data  = total_data_shuffled[:data_training_length, :]
    train_label = total_label_shuffled[:data_training_length, :]

    
    test_data  = total_data_shuffled[data_training_length:, :]
    test_label = total_label_shuffled[data_training_length:, :]

    txt = []
    for no_hidden_layer in list_no_hidden_layer_h:  
        for no_neuron in list_no_unit_in_a_layer_h:  
            for dropout in list_dropout_h:
                print ("no_hidden_layer: %d, no_neuron: %d, dropout: %.2f" % (no_hidden_layer, no_neuron, dropout))
                if args.mode ==  'train':
                    test_relative_err = nn.train_nn(train_data, train_label, test_data, test_label, no_neuron, no_hidden_layer, dropout, model_path)
                    #test_relative_err = nn.test_CV (no_neuron, no_hidden_layer, dropout, model_path)
                #elif args.mode == 'test':
                    #test_relative_err = nn.test_nn(test_data, test_label, no_neuron, no_hidden_layer, model_path)
                    txt.append ((no_hidden_layer, no_neuron, dropout, test_relative_err))
                else:
                    print('mode input is wrong')
    np.savetxt (mean_relative_err_error_file_name, txt, fmt="%d\t%d\t%.2f\t%f")
    stop_time = time.time()
    print ("Total time (s):", stop_time - start_time)





    '''

    def test_nn(self, test_data, test_label, no_neuron, no_hidden_layer, dropout_val, model_path):

        
        """ Start testing, using test data"""
        print ("Start testing!")
        # record the time when testing starts
        start_time = time.time()
    
        #building model
        X, Y, prediction, weights, dropout= self.build_model(self.dim_data, self.dim_label, no_neuron, no_hidden_layer)

        #evaluate model on testing process
        cost = tf.reduce_mean(tf.squared_difference(prediction, Y)) # or can use tf.nn.l2_loss
        rmse = tf.sqrt(cost)

        init = tf.global_variables_initializer()


        with tf.Session() as sess:
            saver = tf.train.Saver()
                                                                     # Model name in the training step may differiate with the testing step due to the argument called 
                                                                     # then, we should use the name of graph stored in /checkpoint/ folder (created after the training step) 
                                                                     # for self.model_name in the testing step.
            
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_path))
            
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print ("Found:", ckpt.model_checkpoint_path)
            else:
                raise ValueError ("Not found model!!!")
            # Score accuracy
            #test_input_fn = tf.estimator.inputs.numpy_input_fn(x=test_data, y=test_label, num_epochs=1, shuffle=False)
            #ev = nn.evaluate(input_fn=test_input_fn)
            test_rmse_val = sess.run(rmse, feed_dict={X: test_data, Y: test_label, dropout:dropout_val})
            #print('test rmse: {:.3f}'.format(test_rmse_val))

            print ("Testing finished!")
            stop_time = time.time()
            print ("Testing time (s):", stop_time - start_time)
            return test_rmse_val

#########################################
        def choose_hyper_param (self, file): 
        """
            - Purpose: Based on the average test error over cross-validation, we have to select which hyper parameter
                will be applied for our model. We will choose which one has the smallest average test error.
            - Input:
            - Return:
        """
        #building model
        X, Y, prediction, dropout, weights = self.build_model(self.dim_data, self.dim_label, self.no_neuron, self.no_hidden_layer)

        #evaluate model on training process
        cost = tf.reduce_mean(tf.squared_difference(prediction, Y)) # or can use tf.nn.l2_loss
        rmse = tf.sqrt(cost)

        init = tf.global_variables_initializer()


        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_file_path)
            
            """ Start training, using training data"""
            # record the time when training starts
            start_time = time.time()
    
            """ In each iteration, we need to store training error and test error into 2 separare lists"""
            test_err = []
            list_no_hidden_layer = list_no_hidden_layer_h
            list_no_unit_in_a_layer = list_no_unit_in_a_layer_h

            no_hidden_layer = list_no_hidden_layer[0]
            no_neuron = list_no_unit_in_a_layer[0]
            
            train_data = self.X_total_set 
            train_label = self.y_total_set
            test_data = self.X_total_set
            test_label = self.y_total_set

            """model_file_path = self.model_dir + '/' + self.model_name # + '_' + str (k_fold) + '_' + str(epoch + 1) + '.ckpt'
                                                                     # Model name in the training step may differiate with the testing step due to the argument called 
                                                                     # then, we should use the name of graph stored in /checkpoint/ folder (created after the training step) 
                                                                     # for self.model_name in the testing step.
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_file_path))
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)"""
            
            train_nn(train_data, train_label, i, no_neuron, no_hidden_layer, model_file_path)

            test_rmse_val = sess.run(rmse, feed_dict={X: test_data, Y: test_label})
            """for i in range (self.k_fold)
                train_data = self.X_train_set[i] 
                train_label = self.y_train_set[i]
                test_data = self.X_test_set[i]
                test_label = self.y_test_set[i]

                model_file_path = self.model_dir + '/' + self.model_name # + '_' + str (k_fold) + '_' + str(epoch + 1) + '.ckpt'
                                                                         # Model name in the training step may differiate with the testing step due to the argument called 
                                                                         # then, we should use the name of graph stored in /checkpoint/ folder (created after the training step) 
                                                                         # for self.model_name in the testing step.
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_file_path))
                # if that checkpoint exists, restore from checkpoint
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                
                train_nn(train_data, train_label, i, no_neuron, no_hidden_layer, model_file_path)

                test_rmse_val = sess.run(rmse, feed_dict={X: test_data, Y: test_label})
            """
            #print test output result'''
