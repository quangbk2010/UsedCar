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
        self.dim_data = args.dim_data
        self.dim_label = args.dim_label
        self.no_hidden_layer = args.no_hidden_layer
        self.neuron_num = args.neuron_num
        self.k_fold = args.k_fold

        Dataset.__init__(self)
        self.features = features

        
        self.feature_constraint = feature_constraint
        self.feature_constraint_values = feature_constraint_values
        self.dropout = dropout_h

        if constraint_flag == 0:
            (self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set) = self.get_data_label (self.features, output)
        else:
            (self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set) = self.get_data_label_with_constraints (self.features, output, self.feature_constraint, self.feature_constraint_values)
        
        self.args = args
        self.dim_label = self.X_total_set.shape[1]

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
        if output == "price":
            y_total_set = self.get_data_array (self.total_dataset, output)
        elif output == "sale_duration":
            y_total_set = self.get_sale_duration_array (self.total_dataset)
        print ("test:", y_total_set)
        X_train_set = []
        y_train_set = []
        X_test_set = []
        y_test_set = []
        
        kf = KFold(self.k_fold)
        
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
        
        X_train_set = []
        y_train_set = []
        X_test_set = []
        y_test_set = []
        
        kf = KFold(self.k_fold)
        
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
            
            
        return (X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set) 
        

    def build_model (self, dim_data, dim_label, no_unit_in_a_hidden_layer, no_hidden_layer, no_epoch, batch_size):
        X = tf.placeholder(tf.float32, [None, dim_data])
        Y = tf.placeholder(tf.float32, [None, dim_label])

        dropout = tf.placeholder(tf.float32, name='dropout')

        W1 = tf.Variable(tf.random_normal([dim_data, no_unit_in_a_hidden_layer], stddev=0.01))
        L1 = tf.nn.relu(tf.matmul(X, W1))

        W2 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer, no_unit_in_a_hidden_layer], stddev=0.01))
        L2 = tf.nn.relu(tf.matmul(L1, W2))
        L2_hat = tf.nn.dropout(L2, dropout, name='relu_dropout')

        W3 = tf.Variable(tf.random_normal([no_unit_in_a_hidden_layer, dim_label], stddev=0.01))
        logits = tf.matmul(L2_hat, W3)

        return X, Y, logits

    def train (self):

        X, Y, logits = self.build_model(self.dim_data, self.dim_label, self.neuron_num, self.no_hidden_layer, self.epoch, self.batch_size)
        
        print (logits.shape)
        
        cost = tf.reduce_mean(tf.squared_difference(logits, Y)) # or can use tf.nn.l2_loss
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

        """ Evaluate model """
        rmse = tf.sqrt(cost)
        accuracy = rmse

        """ Initialize the variables with default values"""
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            """ Start training, using training data"""
            # record the time when training starts
            start_time = time.time()
            
            train_data = self.X_train_set, 
            train_label = self.y_train_set
            test_data = self.X_test_set
            test_label = self.y_test_set
            
            train_set = np.concatenate ((train_data, train_label), axis = 1)
            train_set_shuffled = np.random.permutation(train_set)
            train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
            train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

            total_batch = int(len(self.train_data)/self.batch_size)

            for epoch in range (self.epoch):

                total_cost = 0
                index_counter = 0

                for i in range (total_batch):
                    start_index = index_counter * self.batch_size
                    end_index = (index_counter + 1) * self.batch_size
                    batch_x = train_data_shuffled [start_index : end_index]
                    batch_y = train_label_shuffled [start_index : end_index]
                    
                    index_counter = index_counter + 1

                    if (index_counter >= total_batch):
                        index_counter = 0

                    _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
                    total_cost += cost_val
                    #print ("cost_val:", cost_val)

                print('Epoch: %04d' % (epoch + 1), 'Avg. cost = {:.3f}'.format(total_cost / total_batch))

                # Training data permutation
                train_set_shuffled = np.random.permutation(train_set)
                train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
                train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

            print ("Training finished!")
            stop_time = time.time()
            print ("Training time (s):", stop_time - start_time)
            
            print('accuracy (rmse):', sess.run(accuracy, feed_dict={X: test_data, Y: test_label, dropout:self.DROP_OUT}))
            print ("test predicted hand:", sess.run (logits, feed_dict={X: test_data, dropout:self.DROP_OUT}))
            print ("test ground truth hand:", sess.run (Y, feed_dict={Y: test_label}))
            np.savetxt (mean_rmse_error_file_name, sess.run (accuracy, feed_dict={X: test_data, Y: test_label, dropout:self.DROP_OUT}), fmt = "%f")
              

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #configuration
    parser.add_argument('--gpu_idx', type=str, default = '0')
    parser.add_argument('--mode_name', type=str)
    parser.add_argument('--test', type=bool, default=False)

    #hyper parameter
    parser.add_argument('--epoch', type=int, default = 1)
    parser.add_argument('--batch_size', type=int, default = 100)
    parser.add_argument('--k_fold', type=int, default=5)

    #network parameter
    parser.add_argument('--dim_data', type=int, default=10)
    parser.add_argument('--dim_label', type=int, default=1)
    parser.add_argument('--no_hidden_layer', type=int, default = 2)
    parser.add_argument('--neuron_num', type=int, default = 1000)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx

    task1 = Tensor_NN (args)

    if args.test:
        print('test')
    else:
        task1.train()
