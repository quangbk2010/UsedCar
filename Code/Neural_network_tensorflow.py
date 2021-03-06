#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:41:56 2017

@author: quang
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
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

        if using_car_ident_flag == 1:
            (self.car_ident_code_total_set, self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set, self.d_ident, self.d_remain, self.car_ident_code_test_set) = self.get_data_label_car_ident (self.features, output)
            
        elif constraint_flag == 0:
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
        
        X_total_set = self.get_data_matrix (self.total_dataset, features) 

        len_total_set = X_total_set.shape[0]    
        train_length     = int (0.5 + len_total_set * data_training_percentage)
        #test_length     = int (0.5 + len_total_set * data_test_percentage)

        scaler = StandardScaler()  
        scaler.fit(X_total_set)  
        X_total_set = scaler.transform(X_total_set)  
            

        if output == "price":
            y_total_set = self.get_data_array (self.total_dataset, output)
        elif output == "sale_duration":
            X_total_set = self.get_data_matrix_with_constraint (self.total_dataset, features, "sale_state", "Sold-out") 
            y_total_set = self.get_sale_duration_array (self.total_dataset)
        #print ("test:", self.total_dataset.shape, X_total_set.shape, y_total_set.shape)
        #print (y_total_set[:10])
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
            X_train_set = X_total_set[:train_length, :]
            y_train_set = y_total_set[:train_length, :]

            X_test_set = X_total_set[train_length:, :]
            y_test_set = y_total_set[train_length:, :]

            
        return (X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set) 
    
    
    def get_data_label_car_ident (self, features, output):
        """
            - Purpose: Devide total dataset into n_splits set.
                Using (n_splits - 1) sets for training, the remaining for testing
            - Input:
             + features: an array of feature names
             + output: name of the output (label)
            - Return: a tuple with each element contains a list of "n_splits" training sets (X_train_set, y_train_set) 
             and "n_splits" test sets (X_test_set, y_test_set)
        """
        
        car_ident_code_total_set, X_total_set, d_ident, d_remain = self.get_data_matrix_car_ident (self.get_total_dataset ()) 

        len_total_set = X_total_set.shape[0]    
        train_length     = int (0.5 + len_total_set * data_training_percentage)
        #test_length     = int (0.5 + len_total_set * data_test_percentage)

        scaler = StandardScaler()  
        scaler.fit(X_total_set)  
        X_total_set = scaler.transform(X_total_set)
        """for feature in feature_need_scaler:
            X_total_set[feature] = scaler.fit_transform(X_total_set[feature])  """
            

        if output == "price":
            y_total_set = self.get_data_array (self.total_dataset, output)
        #print ("test:", self.total_dataset.shape, X_total_set.shape, y_total_set.shape)
        #print (y_total_set[:10])
        #sys.exit (-1)
        X_train_set = []
        y_train_set = []
        X_test_set = []
        y_test_set = []
        car_ident_code_test_set = []
        
        if self.k_fold > 0:
            kf = KFold(self.k_fold, shuffle=True)
            
            for train_index, test_index in kf.split(X_total_set):
                
                X_train, X_test = X_total_set[train_index], X_total_set[test_index]
                y_train, y_test = y_total_set[train_index], y_total_set[test_index]
                test_car_ident_codes = car_ident_code_total_set[test_index]
               
                X_train_set.append (X_train)
                y_train_set.append (y_train)
                X_test_set.append (X_test)
                y_test_set.append (y_test)
                car_ident_code_test_set.append (test_car_ident_codes)

        else:
            X_train_set = X_total_set[:train_length, :]
            y_train_set = y_total_set[:train_length, :]

            X_test_set = X_total_set[train_length:, :]
            y_test_set = y_total_set[train_length:, :]
            car_ident_code_test_set = car_ident_code_total_set[train_length:, :]

            
        return (car_ident_code_total_set, X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set, d_ident, d_remain, car_ident_code_test_set) 

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

        len_total_set = X_total_set.shape[0]    
        train_length     = int (0.5 + len_total_set * data_training_percentage)
        #test_length     = int (0.5 + len_total_set * data_test_percentage)
        
           
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
            X_train_set = X_total_set[:train_length, :]
            y_train_set = y_total_set[:train_length, :]

            X_test_set = X_total_set[train_length:, :]
            y_test_set = y_total_set[train_length:, :]
            
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

    def build_car2vect_model (self, no_neuron, no_neuron_embed, d_ident, d_embed, d_remain):
        """
            - Args:
                + no_neuron: the number of neuron in 'main' hiddenlayers.
                + no_neuron_embed: the number of neuron in 'embedding' hiddenlayers.
                + d_ident: the dimension of the one-hot vector of a car identification (manufacture_code, ..., rating_code)
                + d_remain: the dimension of the remaining features (after one-hot encoding)
            - Purpose: Create the model of NN with using car2vect: from the one-hot encode of a car identification, use word2vect to embed it into a vector with small dimension.
        """
    
        #tf.reset_default_graph() 

        x_ident = tf.placeholder(tf.float32, [None, d_ident])
        x_remain = tf.placeholder(tf.float32, [None, d_remain])
        Y = tf.placeholder(tf.float32, [None, 1])
        dropout = tf.placeholder(tf.float32, name='dropout')

        """output1 = slim.fully_connected(x_ident, no_neuron_embed, scope='hidden1', activation_fn=tf.nn.relu)
        x_embed = slim.fully_connected(output1, d_embed, scope='hidden2', activation_fn=tf.nn.relu) # 3-dimension of embeding NN

        input3 = tf.concat ([x_remain, x_embed], 1)

        output3 = slim.fully_connected(input3, no_neuron, scope='hidden3', activation_fn=tf.nn.relu)
        prediction = slim.fully_connected(output3, 1, scope='output') # 1-dimension of output"""
        
        print ("build_car2vect_model: d_ident:", d_ident, "d_remain:", d_remain, "d_embed:", d_embed, "no_neuron_embed:", no_neuron_embed, "no_neuron_main:", no_neuron)

        #output1 = slim.fully_connected(x_ident, d_ident, scope='input_embed', activation_fn=tf.nn.relu)
        output1 = slim.fully_connected(x_ident, no_neuron_embed, scope='input_embed', activation_fn=tf.nn.relu)
        #output1 = slim.dropout(output1, 0.5, scope='dropout1')
        #output2 = slim.fully_connected(output1, no_neuron_embed, scope='hidden_embed1', activation_fn=tf.nn.relu)
        #output3 = slim.fully_connected(output2, no_neuron_embed, scope='hidden_embed2', activation_fn=tf.nn.relu)
        x_embed = slim.fully_connected(output1, d_embed, scope='output_embed', activation_fn=tf.nn.relu) # 3-dimension of embeding NN

        input3 = tf.concat ([x_remain, x_embed], 1)

        #output3 = slim.fully_connected(input3, d_remain + d_embed, scope='input_main', activation_fn=tf.nn.relu)
        output3 = slim.fully_connected(input3, no_neuron, scope='input_main', activation_fn=tf.nn.relu)
        #output4 = slim.fully_connected(output3, no_neuron, scope='hidden_main_1', activation_fn=tf.nn.relu)
        #output5 = slim.fully_connected(output4, no_neuron, scope='hidden_main_2', activation_fn=tf.nn.relu)
        prediction = slim.fully_connected(output3, 1, scope='output_main') # 1-dimension of output


        return x_ident, x_remain, Y, x_embed, prediction, dropout

    def car2vect(self, train_data, train_label, test_data, test_label, test_car_ident, no_neuron, dropout_val, model_path, d_ident, d_embed, d_remain, no_neuron_embed): # Used for 1train-1test
    #def car2vect(self, train_data, train_label, test_data, test_label, test_car_ident, dropout_val, model_path, d_ident, d_embed, d_remain, x_ident, x_remain, Y, x_embed, prediction, dropout, fold): # used for Cross-validation 

        #building car embedding model
        if using_CV_flag == 0:
            x_ident, x_remain, Y, x_embed, prediction, dropout = self.build_car2vect_model(no_neuron, no_neuron_embed, d_ident, d_embed, d_remain)
            x_ident_file_name_ = x_ident_file_name
            x_embed_file_name_ = x_embed_file_name
            mean_error_file_name_ = mean_error_file_name
            y_predict_file_name_ = y_predict_file_name
        else:
            x_ident_file_name_ = x_ident_file_name + "_" + "fold" + str (fold+1)
            x_embed_file_name_ = x_embed_file_name + "_" + "fold" + str (fold+1)
            mean_error_file_name_ = mean_error_file_name + "_" + "fold" + str (fold+1)
            y_predict_file_name_ = y_predict_file_name + "_" + "fold" + str (fold+1)


        # Try to use weights decay
        num_batches_per_epoch = int(len(train_data) / self.batch_size)
        decay_steps = int(num_batches_per_epoch * self.decay_step)
        global_step = tf.Variable(0, trainable = False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, decay_steps, self.decay_rate, staircase = True) # 0.00125#

        # Used for minimizing relative error
        loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), Y)) 
        #loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) ))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
        # RMSE
        rmse = tf.sqrt (tf.reduce_mean(tf.squared_difference(prediction, Y)))
        mae = tf.reduce_mean (tf.abs (prediction - Y))
        relative_err = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), Y)) * 100 # there are problem when Y = 0 -> inf or nan answer
        smape = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) )) * 100
        
        # Calculate root mean squared error as additional eval metric

        #""" Initialize the variables with default values"""
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

            train_data_remain_shuffled = train_set_shuffled [:, 0:d_remain]
            train_data_ident_shuffled = train_set_shuffled [:, d_remain:train_data.shape[1]]
            train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

            test_data_remain = test_data [:, 0:d_remain]
            test_data_ident = test_data [:, d_remain:]

            print ("test car ident", test_car_ident[0])
            np.savetxt (x_ident_file_name_, test_car_ident, fmt="%d\t%d\t%d\t%d\t%s")  #[manufacture_codes,rep_model_codes,car_codes,model_codes,rating_codes, x_embed_val]
            
            print ("len train_data_ident:", train_data_ident_shuffled.shape)
            print ("len train_data_remain:", train_data_remain_shuffled.shape)
            print ("len train_label:", train_label_shuffled.shape)

            print ("train_data_ident[0]:", train_data_ident_shuffled[0])
            print ("train_data_remain[0]:", train_data_remain_shuffled[0])
            print ("train_label[0]:", train_label_shuffled[0])
            sys.exit (-1)


            total_batch = int((len(train_data)/self.batch_size) + 0.5)

            pre_epoch_test_relative_err_val = 0

            epoch_list = [] 
            train_err_list = [] 
            rmse_list = []
            mae_list = []
            rel_err_list = []
            smape_list = []

            for epoch in range (self.epoch):

                total_rmse = 0
                total_mae = 0
                total_relative_err = 0
                total_smape = 0
                index_counter = 0
                left_num = len(train_data)
                for i in range (total_batch):
                    start_index = index_counter * self.batch_size
                    #end_index = 0
                    if(left_num < self.batch_size):
                        end_index = index_counter * self.batch_size + left_num
                    else:
                        end_index = (index_counter + 1) * self.batch_size

                    batch_x_ident = train_data_ident_shuffled [start_index : end_index]
                    batch_x_remain = train_data_remain_shuffled [start_index : end_index]
                    batch_y = train_label_shuffled [start_index : end_index]
                    #print ("batch_x", batch_x)
                    #print ("batch_y", batch_y)
                    index_counter = index_counter + 1
                    left_num = left_num - self.batch_size

                    if (left_num <= 0):
                        index_counter = 0
# x_ident, x_remain, Y,
                    _, training_rmse_val, training_mae_val, training_relative_err_val, training_smape_val = sess.run([optimizer, rmse, mae, relative_err, smape], feed_dict={x_ident: batch_x_ident, x_remain: batch_x_remain, Y: batch_y, dropout:dropout_val})
                    total_rmse += training_rmse_val
                    total_mae += training_mae_val
                    total_relative_err += training_relative_err_val
                    total_smape += training_smape_val
                    
                    #a, b = sess.run([Y, prediction], feed_dict={x_ident: batch_x_ident, x_remain: batch_x_remain, Y: batch_y, dropout:dropout_val})
                    #print ("truth:", a[0][0], "prediction:", b[0][0], "Err:", training_relative_err_val)

                #print('Epoch: %04d' % (epoch + 1), 'Avg. rmse = {:.3f}'.format(total_rmse / total_batch), 'learning_rate = {:.5f}'.format(lr))
                print('\n\nEpoch: %04d' % (epoch + 1), "Avg. training rmse:", total_rmse/total_batch, "mae:", total_mae/total_batch, 'relative_err:', total_relative_err/total_batch, "smape:", total_smape/total_batch)

                #sys.exit (-1)
                
                predicted_y, x_embed_val, epoch_test_rmse_val, epoch_test_mae_val, epoch_test_relative_err_val, epoch_test_smape_val = sess.run([prediction, x_embed, rmse, mae, relative_err, smape], feed_dict={x_ident: test_data_ident, x_remain: test_data_remain, Y: test_label, dropout:self.dropout})
                
                print ("test: rmse", epoch_test_rmse_val)
                print ("test: mae", epoch_test_mae_val)
                print ("test: relative_err", epoch_test_relative_err_val)
                print ("test: smape", epoch_test_smape_val)
                print ("truth:", test_label[:10], "prediction:", predicted_y[:10])

                epoch_list.append (epoch)
                train_err_list.append (total_relative_err/total_batch)
                rmse_list.append (epoch_test_rmse_val)
                mae_list.append (epoch_test_mae_val)
                rel_err_list.append (epoch_test_relative_err_val)
                smape_list.append (epoch_test_smape_val)

                np.savetxt (x_embed_file_name_ + "_" + str (epoch), x_embed_val, fmt="%.2f\t%.2f\t%.2f")

                line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
                line['truth'] = test_label.reshape (test_label.shape[0])
                line['pred'] = predicted_y.reshape (predicted_y.shape[0])
                np.savetxt(y_predict_file_name_ + "_" + str (epoch), line, fmt="%.2f\t%.2f")

                #sys.exit (-1)
                #if epoch_test_relative_err_val > pre_epoch_test_relative_err_val and pre_epoch_test_relative_err_val != 0:
                    #break
                #pre_epoch_test_relative_err_val = epoch_test_relative_err_val

                #TODO: training data permutation
                train_set_shuffled = np.random.permutation(train_set)
                train_data_remain_shuffled = train_set_shuffled [:, 0:d_remain]
                train_data_ident_shuffled = train_set_shuffled [:, d_remain:train_data.shape[1]]
                train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]


                """if (epoch + 1) % self.saved_period == 0 and epoch != 0:
                    #model_path = self.model_dir + '/' + self.model_name + '_' + str (k_fold) + '_' + str(epoch + 1) + '.ckpt'
                    save_path = saver.save(sess, model_path, global_step=global_step)
                    print('Model saved in file: %s' % save_path)"""

            print ("Training finished!")
            stop_time = time.time()
            print ("Training time (s):", stop_time - start_time)
            
            print('test last epoch rel_err: {:.3f}'.format(epoch_test_relative_err_val))

            line = np.zeros(len (epoch_list), dtype=[('epoch', int), ('rmse', float), ('mae', float), ('rel_err', float), ('smape', float), ('train_rel_err', float)])
            line['epoch'] = epoch_list
            line['rmse'] = rmse_list
            line['mae'] = mae_list
            line['rel_err'] = rel_err_list
            line['smape'] = smape_list
            line['train_rel_err'] = train_err_list
            np.savetxt(mean_error_file_name_ + "_" + str (epoch), line, fmt="%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f")

            return epoch_test_relative_err_val

    #def train_nn(self, train_data, train_label, test_data, test_label, no_neuron, no_hidden_layer, dropout_val, model_path): # Used for 1train-1test
    def train_nn(self, train_data, train_label, test_data, test_label, dropout_val, model_path, X, Y, prediction, weights, dropout, fold): # used for Cross-validation 
       
        #building car embedding model
        if using_CV_flag == 0:
            X, Y, prediction, weights, dropout = self.build_model(train_data.shape[1], self.dim_label, no_neuron, no_hidden_layer) 
            mean_error_file_name_ = mean_error_file_name
            y_predict_file_name_ = y_predict_file_name
        else:
            mean_error_file_name_ = mean_error_file_name + "_" + "fold" + str (fold+1)
            y_predict_file_name_ = y_predict_file_name + "_" + "fold" + str (fold+1)


        # Try to use weights decay
        num_batches_per_epoch = int(len(train_data) / self.batch_size)
        decay_steps = int(num_batches_per_epoch * self.decay_step)
        global_step = tf.Variable(0, trainable = False)
        #learning_rate = self.learning_rate
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, decay_steps, self.decay_rate, staircase = True)

        # Used for minimizing RMSE
        #loss = tf.reduce_mean(tf.squared_difference(prediction, Y))

        # Used for minimizing relative error
        loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), Y)) #protect when Y = 0

        # Used for minimizing SMAPE
        #loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) ))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
        # RMSE
        rmse = tf.sqrt (tf.reduce_mean(tf.squared_difference(prediction, Y)))
        mae = tf.reduce_mean (tf.abs (prediction - Y))
        relative_err = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), (Y))) * 100 # there are problem when Y = 0 -> inf or nan answer
        smape = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) )) * 100
        
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

            #print ("len train:", train_data.shape)
            total_batch = int((len(train_data)/self.batch_size) + 0.5)

            pre_epoch_test_relative_err_val = 0

            epoch_list = [] 
            train_err_list = [] 
            rmse_list = []
            mae_list = []
            rel_err_list = []
            smape_list = []

            for epoch in range (self.epoch):

                total_rmse = 0
                total_mae = 0
                total_relative_err = 0
                total_smape = 0
                index_counter = 0
                left_num = len(train_data)
                for i in range (total_batch):
                    start_index = index_counter * self.batch_size
                    #end_index = 0
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
                    _, training_rmse_val, training_mae_val, training_relative_err_val, training_smape_val = sess.run([optimizer, rmse, mae, relative_err, smape], feed_dict={X: batch_x, Y: batch_y, dropout:dropout_val})
                    total_rmse += training_rmse_val
                    total_mae += training_mae_val
                    total_relative_err += training_relative_err_val
                    total_smape += training_smape_val
                    
                    a, b = sess.run([Y, prediction], feed_dict={X: batch_x, Y: batch_y, dropout:dropout_val})
                    #print ("truth:", a[0][0], "prediction:", b[0][0], "Err:", training_relative_err_val)

                #print('Epoch: %04d' % (epoch + 1), 'Avg. rmse = {:.3f}'.format(total_rmse / total_batch), 'learning_rate = {:.5f}'.format(lr))
                print('\n\nEpoch: %04d' % (epoch + 1), "Avg. training rmse:", total_rmse/total_batch, "mae:", total_mae/total_batch, 'relative_err:', total_relative_err/total_batch, "smape:", total_smape/total_batch)
                
                predicted_y, epoch_test_rmse_val, epoch_test_mae_val, epoch_test_relative_err_val, epoch_test_smape_val = sess.run([prediction, rmse, mae, relative_err, smape], feed_dict={X: test_data, Y: test_label, dropout:self.dropout})
                
                print ("test: rmse", epoch_test_rmse_val)
                print ("test: mae", epoch_test_mae_val)
                print ("test: relative_err", epoch_test_relative_err_val)
                print ("test: smape", epoch_test_smape_val)
                print ("truth:", test_label[:10], "prediction:", predicted_y[:10])

                epoch_list.append (epoch)
                train_err_list.append (total_relative_err/total_batch)
                rmse_list.append (epoch_test_rmse_val)
                mae_list.append (epoch_test_mae_val)
                rel_err_list.append (epoch_test_relative_err_val)
                smape_list.append (epoch_test_smape_val)

                line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
                line['truth'] = test_label.reshape (test_label.shape[0])
                line['pred'] = predicted_y.reshape (predicted_y.shape[0])
                np.savetxt(y_predict_file_name_ + "_" + str (epoch), line, fmt="%.2f\t%.2f")


                #TODO: training data permutation
                train_set_shuffled = np.random.permutation(train_set)
                train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
                train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

                """if (epoch + 1) % self.saved_period == 0 and epoch != 0:
                    #model_path = self.model_dir + '/' + self.model_name + '_' + str (k_fold) + '_' + str(epoch + 1) + '.ckpt'
                    save_path = saver.save(sess, model_path, global_step=global_step)
                    print('Model saved in file: %s' % save_path)"""

            print ("Training finished!")
            stop_time = time.time()
            print ("Training time (s):", stop_time - start_time)
            
            print('test last epoch rel_err: {:.3f}'.format(epoch_test_relative_err_val))

            line = np.zeros(len (epoch_list), dtype=[('epoch', int), ('rmse', float), ('mae', float), ('rel_err', float), ('smape', float), ('train_rel_err', float)])
            line['epoch'] = epoch_list
            line['rmse'] = rmse_list
            line['mae'] = mae_list
            line['rel_err'] = rel_err_list
            line['smape'] = smape_list
            line['train_rel_err'] = train_err_list
            np.savetxt(mean_error_file_name_ + "_" + str (epoch), line, fmt="%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f")

            return epoch_test_relative_err_val
   
    
    #def test_CV (self, no_neuron, no_hidden_layer, dropout_val, model_path):
    def test_CV (self, model_path):
        
        with tf.Session() as sess:
            """ In each iteration, we need to store training error and test error into 2 separare lists"""
            print ("Start k-fold CV:")
            d_embed_val = 3
            if using_car_ident_flag == 1:
                print ("--Build model using car identification")
                x_ident, x_remain, Y, x_embed, prediction, dropout = self.build_car2vect_model(no_neuron=100, no_neuron_embed=6000, d_ident=nn.d_ident,d_embed=d_embed_val, d_remain=nn.d_remain)
            else:
                print ("--Build model using only onehot code")
                X, Y, prediction, weights, dropout = self.build_model(dim_data=self.X_train_set[0].shape[1], dim_label=self.dim_label, no_unit_in_a_hidden_layer=6000, no_hidden_layer=2)

            
            test_err = []
            for i in range (self.k_fold):
                print ("Start fold:", i)
                train_data = self.X_train_set[i] 
                train_label = self.y_train_set[i]
                test_data = self.X_test_set[i]
                test_label = self.y_test_set[i]
                if using_car_ident_flag == 1:
                    test_car_ident = self.car_ident_code_test_set[i]

                #print ("train_data.shape", train_data.shape,"train_label.shape",  train_label.shape, "test_data.shape", test_data.shape, "test_data.shape", test_label.shape, "test_car_ident.shape", test_car_ident.shape)
                #print ("train_data[0]", train_data[0], "train_label[0:10]", train_label[0:10], "test_data[0]", test_data[0], "test_label[0:10]", test_label[0:10], "test_car_ident[0]", test_car_ident[0])
                #sys.exit (-1)
                
                
                train_set = np.concatenate ((train_data, train_label), axis = 1)
                train_set_shuffled = np.random.permutation(train_set)
                train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
                train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

                if using_car_ident_flag == 1:
                    test_relative_err_val = self.car2vect(train_data=train_data_shuffled, train_label=train_label_shuffled, test_data=test_data, test_label=test_label, test_car_ident=test_car_ident, dropout_val=1, model_path=model_path, d_ident=self.d_ident,d_embed=d_embed_val, d_remain=self.d_remain, x_ident=x_ident, x_remain=x_remain, Y=Y, x_embed=x_embed, prediction=prediction, dropout=dropout, fold=i) # 1000, 3, 6000
                else:
                    test_relative_err_val = self.train_nn (train_data=train_data_shuffled, train_label=train_label_shuffled, test_data=test_data, test_label=test_label, dropout_val=1, model_path=model_path, X=X, Y=Y, prediction=prediction, weights=weights, dropout=dropout, fold=i)

                test_err.append (test_relative_err_val) 
                print ("Fold:", i, test_relative_err_val)
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
    parser.add_argument('--epoch', type=int, default = 50) #2000 # 100
    parser.add_argument('--dropout', type=int, default = 1)
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--learning_rate', type=float, default=0.0125)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--decay_step', type=int, default=100) #if decay_step > epoch, no exponential decay

    #network parameter
    parser.add_argument('--dim_data', type=int, default=24)
    parser.add_argument('--dim_label', type=int, default=1)
    parser.add_argument('--no_hidden_layer', type=int, default = 1) #not implement variabel network layer
    parser.add_argument('--no_neuron', type=int, default = 1000)
    parser.add_argument('--k_fold', type=int, default = -1) # set it to -1 when don't want to use k-fold CV

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx

    # record the time when training starts
    start_time = time.time()

    
    nn = Tensor_NN (args)


    model_path = nn.model_dir + '/' + nn.model_name

    print ("Shape:", nn.X_total_set.shape, nn.y_total_set.shape)
    if using_CV_flag == 1:
        print ("Test CV:")
        if nn.k_fold > 0:
            nn.test_CV (model_path)
        else:
            print ("kfold < 0")
        sys.exit (-1)

    # otherwise, used for 1train-1test
    print ("1train-1test:")
    if nn.k_fold > 0:
        print ("kfold > 0")
        sys.exit (-1)

    if using_shuffle_flag == 1:
        concate_car_ident = np.concatenate ((nn.y_total_set, nn.car_ident_code_total_set), axis = 1) # car_ident_codes used only to draw embeded vector
        total_set_car_ident = np.concatenate ((nn.X_total_set, concate_car_ident), axis = 1)
        total_set = np.concatenate ((nn.X_total_set, nn.y_total_set), axis = 1)

        len_total_set = total_set.shape[0]   
        dim_data_label = total_set.shape[1]
        dim_data_label_car_ident = total_set_car_ident.shape[1]

        #print ("data before suffling:", total_set_car_ident[:5, dim_data_label:])
        #print ("label before suffling:", nn.y_total_set[:])
        #np.savetxt (total_set_shuffled_file_name + "_before", nn.y_total_set, fmt="%f")
        train_length     = int (0.5 + len_total_set * data_training_percentage)
        test_length     = int (0.5 + len_total_set * data_test_percentage)

        total_set_shuffled = np.random.permutation(total_set_car_ident)
        total_data_shuffled = total_set_shuffled [:, 0:dim_data_label-1]
        total_label_shuffled = total_set_shuffled [:, dim_data_label-1:dim_data_label]

        #print ("data after suffling:", total_set_shuffled[:5, dim_data_label:])
        #print ("label after suffling:", total_label_shuffled[:])
        #np.savetxt (total_set_shuffled_file_name, total_label_shuffled, fmt="%f")

    if add_noise_flag == 1:
        # Creating a noise with the same dimension, and add to price
        SD = 5
        mu, sigma = 0, 5
        generator = nn.get_truncated_normal(mean=0.1, sd=0.1, low=-100000, upp=100000)
        alpha = generator.rvs (total_set.shape[0]).reshape (total_set.shape[0], 1)
        print ("alpha noise: ", "min:", np.min (alpha), "max:", np.max (alpha), "mean:", np.mean (alpha))
        total_label_shuffled *= (1-alpha)

    
    if using_shuffle_flag == 1:
        train_data  = total_data_shuffled[:train_length, :]
        train_label = total_label_shuffled[:train_length, :]
        
        test_data  = total_data_shuffled[train_length:, :]
        test_label = total_label_shuffled[train_length:, :]
        test_car_ident = total_set_shuffled [train_length:, dim_data_label:] # car identification
        np.savetxt (train_set_file_name, np.concatenate ((train_label, total_set_shuffled [:train_length, dim_data_label:]), axis=1), fmt = "%.2f\t%d\t%d\t%d\t%d\t%s")
        np.savetxt (test_set_file_name, np.concatenate ((test_label, test_car_ident), axis=1), fmt = "%.2f\t%d\t%d\t%d\t%d\t%s")
    else:
        train_data = nn.X_train_set
        train_label = nn.y_train_set
        test_data = nn.X_test_set
        test_label = nn.y_test_set
        if using_car_ident_flag == 1:
            test_car_ident = nn.car_ident_code_total_set[train_data.shape[0]:]

    print ("train_data:", train_data.shape)
    print ("train_label:", train_label.shape)
    print ("test_data:", test_data.shape)
    print ("test_label:", test_label.shape)
    print ("train_data:", train_data.shape)
 
    if using_CV_flag == 0:
        if using_car_ident_flag == 1:
            nn.car2vect(train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, test_car_ident=test_car_ident, no_neuron=1000, dropout_val=1, model_path=model_path, d_ident=nn.d_ident,d_embed=3, d_remain=nn.d_remain, no_neuron_embed=10000) # 1000, 3, 6000
        else:
            test_relative_err = nn.train_nn(train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, no_neuron=6000, no_hidden_layer=2, dropout_val=1, model_path=model_path)
         
    
    sys.exit (-1)

    
    # These lines below used for choosing hyper-param
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
    np.savetxt (mean_err_file_name, txt, fmt="%d\t%d\t%.2f\t%f")
    stop_time = time.time()
    print ("Total time (s):", stop_time - start_time)


