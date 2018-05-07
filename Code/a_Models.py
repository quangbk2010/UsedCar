
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:00:00 2017

@author: quangbk2010
"""
from a_Header import *
from a_Dataset import *
from a_Performance import *

"""
    ########################
    ### SHORT-LIST MODELS
    ########################
     - Implement, apply algorithms, build models: 
        + Linear, Ridge, Lasso
        + Decision tree: "DecisionTreeRegressor", "GradientBoostingRegressor", "RandomForestRegressor", "AdaBoostRegressor"
        + Deep learning: baseline, car2vec
        + Ensemble methods: 
"""
        
class Tensor_NN (Dataset):

    def __init__(self, args):
        #from args
        self.gpu_idx = args.gpu_idx

        self.test_uncertainty = args.test_uncertainty

        self.epoch1 = args.epoch1
        self.epoch2 = args.epoch2
        self.restored_epoch = args.restored_epoch
        self.batch_size = args.batch_size
        self.dropout = args.dropout

        self.learning_rate = args.learning_rate
        self.decay_step = args.decay_step
        self.decay_rate = args.decay_rate
        self.saved_period = args.saved_period

        self.d_embed = args.d_embed
        self.no_neuron = args.no_neuron
        self.no_neuron_embed = args.no_neuron_embed
        self.loss_func = args.loss_func

        self.model_dir = args.model_dir
        self.args = args

    def build_car2vec_model (self, no_neuron, no_neuron_embed, d_ident, d_embed, d_remain):
        """
            - Args:
                + no_neuron: the number of neuron in 'main' hiddenlayers.
                + no_neuron_embed: the number of neuron in 'embedding' hiddenlayers.
                + d_ident: the dimension of the one-hot vector of a car identification (manufacture_code, ..., rating_code)
                + d_remain: the dimension of the remaining features (after one-hot encoding)
            - Purpose: Create the model of NN with using car2vec: from the one-hot encode of a car identification, use word2vect to embed it into a vector with small dimension.
        """
        x_ident = tf.placeholder(tf.float32, [None, d_ident], name="x_ident")
        x_remain = tf.placeholder(tf.float32, [None, d_remain], name="x_remain")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        print ("build_car2vec_model: d_ident:{0}, d_remain:{1}, d_embed:{2}, no_neuron_embed:{3}, no_neuron_main:{4}".format (d_ident, d_remain, d_embed, no_neuron_embed, no_neuron))
        
        # He initializer
        he_init = tf.contrib.layers.variance_scaling_initializer ()
        output1 = slim.fully_connected (x_ident, no_neuron_embed, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=he_init) 
        output1_ = slim.dropout (output1, self.dropout, scope='dropout1')
        x_embed = slim.fully_connected (output1_, d_embed, scope='output_embed', activation_fn=None, weights_initializer=he_init) 
        input3 = tf.concat ([x_remain, x_embed], 1)

        output3 = slim.fully_connected(input3, no_neuron, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=he_init)
        output3_ = slim.dropout (output3, self.dropout, scope='dropout3')
        prediction = slim.fully_connected(output3_, 1, scope='output_main', activation_fn=tf.nn.relu, weights_initializer=he_init) 
        tf.identity (prediction, name="prediction")

        return x_ident, x_remain, Y, x_embed, prediction, phase_train

    def restore_model_car2vec (self, sess, x_ident_, x_remain_, label, meta_file, ckpt_file, train_flag):
        # Access and create placeholder variables, and create feedict to feed data 
        # Get the default graph for the current thread
        graph = tf.get_default_graph ()
        x_ident = graph.get_tensor_by_name ("x_ident:0")
        x_remain = graph.get_tensor_by_name ("x_remain:0")
        Y = graph.get_tensor_by_name ("Y:0")
        feed_dict = {x_ident:x_ident_, x_remain:x_remain_, Y:label}

        # Now, access the operators that we want to run
        prediction = graph.get_tensor_by_name ("prediction:0")

        # And feed data
        if train_flag == 0:
            rmse= graph.get_tensor_by_name ("rmse:0")
            mae = graph.get_tensor_by_name ("mae:0")
            rel_err = graph.get_tensor_by_name ("rel_err:0")
            smape = graph.get_tensor_by_name ("smape:0")

            predicted_y, rmse_val, mae_val, rel_err_val, smape_val = sess.run([prediction, rmse, mae, rel_err, smape], feed_dict)
            return (predicted_y, rmse_val, mae_val, rel_err_val, smape_val)
        else:
            sum_se= graph.get_tensor_by_name ("sum_se:0")
            sum_ae = graph.get_tensor_by_name ("sum_ae:0")
            sum_rel_err = graph.get_tensor_by_name ("sum_rel_err:0")
            arr_rel_err = graph.get_tensor_by_name ("arr_rel_err:0")
            sum_smape = graph.get_tensor_by_name ("sum_smape:0")

            predicted_y, sum_se_val, sum_ae_val, sum_rel_err_val, sum_smape_val, arr_rel_err_val = sess.run([prediction, sum_se, sum_ae, sum_rel_err, sum_smape, arr_rel_err], feed_dict)
            return (predicted_y, sum_se_val, sum_ae_val, sum_rel_err_val, sum_smape_val, arr_rel_err_val)


    def train_valid_car2vec (self, train_data, train_label, test_data, test_label, total_car_ident, total_act_adv_date, total_sale_date, d_ident, d_embed, d_remain, no_neuron, no_neuron_embed, loss_func, model_path, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, retrain, nu_epoch): # Used for 1train-1test, add total_act_adv_date, sale_date

        #building car embedding model
        len_train = len(train_data)
        len_test  = len(test_data)
        d_data = train_data.shape[1]

        train_car_ident = total_car_ident [:len_train, :] 
        test_car_ident = total_car_ident [len_train:, :] 
        test_act_adv_date = total_act_adv_date [len_train:, :] 
        test_sale_date = total_sale_date [len_train:, :] 
        test_data_remain = test_data [:, 0:d_remain]
        test_data_ident = test_data [:, d_remain:]


        if retrain == -1:
            print ("=====Restore the trained model...")
            # Apply the trained model onto the test data
            meta_file = model_path + ".meta"
            ckpt_file = model_path 
            
            (predicted_test_label, test_rmse_val, test_mae_val, test_rel_err_val, test_smape_val, test_arr_rel_err) = self.batch_computation_car2vec (5, test_data, test_label, d_ident, d_remain, meta_file, ckpt_file, 1)

            # Save the identification and results into a file
            result = np.concatenate ((test_car_ident, test_act_adv_date, test_sale_date, test_label, predicted_test_label, std),axis=1)
            np.savetxt (x_ident_file_name + "_restored", result, fmt="%d\t%d\t%d\t%d\t%s\t%s\t%d\t%d\t%.2f\t%.2f")  

            return 0 
            
        elif retrain == 0:
            x_ident, x_remain, Y, x_embed, prediction, phase_train = self.build_car2vec_model(no_neuron, no_neuron_embed, d_ident, d_embed, d_remain)
            start_time = time.time()
            print ("========== Time for building the new model:{0}".format (time.time () - start_time))

        x_ident_file_name_ = x_ident_file_name
        x_embed_file_name_ = x_embed_file_name
        mean_error_file_name_ = mean_error_file_name
        y_predict_file_name_ = y_predict_file_name

        identification = np.concatenate ((test_car_ident, test_act_adv_date, test_sale_date),axis=1)
        np.savetxt (x_ident_file_name_, identification, fmt="%d\t%d\t%d\t%d\t%s\t%s\t%d")  

        # Use weights decay
        num_batches_per_epoch = int(len(train_data) / self.batch_size)
        decay_steps = int(num_batches_per_epoch * self.decay_step)
        global_step = tf.Variable(0, trainable = False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, decay_steps, self.decay_rate, staircase = True)

        # Used for minimizing relative error
        loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), Y)) #protect when Y = 0
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step) # + regul
    
        # Declare error functions
        sum_se = tf.reduce_sum (tf.squared_difference(prediction, Y), name = "sum_se")
        sum_ae = tf.reduce_sum (tf.abs (prediction - Y), name = "sum_ae")
        #sum_rel_err = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y), tf.maximum (Y, prediction))), 100, name = "sum_rel_err")
        sum_rel_err = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y), Y)), 100, name = "sum_rel_err")
        sum_smape = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) )), 100, name = "sum_smape")

        arr_rel_err = tf.multiply (tf.divide (tf.abs (prediction - Y), Y), 100, name = "arr_rel_err")

        rmse = tf.sqrt (tf.reduce_mean(tf.squared_difference(prediction, Y)), name = "rmse")
        mae = tf.reduce_mean (tf.abs (prediction - Y), name = "mae")
        #rel_err = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.maximum (Y, prediction))), 100, name = "rel_err") 
        rel_err = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y), Y)), 100, name = "rel_err") 
        smape = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) )), 100, name = "smape")

        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=100)

            # Run the initializer
            sess.run(init)

            """ Start training, using training data"""
            # record the time when training starts
            start_time = time.time()

            train_set = np.concatenate ((train_data, train_label, train_car_ident), axis = 1)
            train_set_shuffled = np.random.permutation(train_set)
            train_data_remain_shuffled = train_set_shuffled [:, 0:d_remain]
            train_data_ident_shuffled = train_set_shuffled [:, d_remain:d_data]
            train_label_shuffled = train_set_shuffled [:, d_data:d_data+1]
            train_car_ident_shuffled = train_set_shuffled [:, d_data+1:]

            print ("len train_data_ident:{0}".format (train_data_ident_shuffled.shape))
            print ("len train_data_remain:{0}".format (train_data_remain_shuffled.shape))
            print ("len train_label:{0}".format (train_label_shuffled.shape))
            print ("len test_data:{0}".format (test_data.shape))

            train_total_batch = int (np.ceil (float (len_train)/self.batch_size))
            test_total_batch = int (np.ceil (float (len_test)/self.batch_size))
            smallest_epoch_test_err_val = 10e6

            epoch_list = [] 
            train_rmse_list = [] 
            train_mae_list = [] 
            train_rel_err_list = [] 
            train_smape_list = [] 
            rmse_list = []
            mae_list = []
            rel_err_list = []
            smape_list = []

            #N = tqdm.trange (nu_epoch)
            #for epoch in range (nu_epoch):
            for epoch in tqdm (range (nu_epoch)):

                # Train the model.
                total_se = 0
                total_ae = 0
                total_rel_err = 0
                total_smape = 0
                start_index = 0
                end_index = 0
                for i in range (train_total_batch):
                    if len_train - end_index < self.batch_size:
                        end_index = len_train
                    else:
                        end_index = (i+1) * self.batch_size

                    batch_x_ident = train_data_ident_shuffled [start_index : end_index]
                    batch_car_ident = train_car_ident_shuffled [start_index : end_index]
                    batch_x_remain = train_data_remain_shuffled [start_index : end_index]
                    batch_y = train_label_shuffled [start_index : end_index]

                    start_index = end_index

                    _, train_sum_se_val, train_sum_ae_val, train_sum_rel_err_val, train_sum_smape_val = sess.run([optimizer, sum_se, sum_ae, sum_rel_err, sum_smape], feed_dict={x_ident: batch_x_ident, x_remain: batch_x_remain, Y: batch_y, phase_train: True})
                    total_se += train_sum_se_val
                    total_ae += train_sum_ae_val
                    total_rel_err += train_sum_rel_err_val
                    total_smape += train_sum_smape_val

                assert end_index == len_train   

                epoch_train_rmse_val = np.sqrt (total_se/len_train)
                epoch_train_mae_val = total_ae/len_train
                epoch_train_rel_err_val = total_rel_err/len_train
                epoch_train_smape_val = total_smape/len_train
                print ('\n\nEpoch: %04d, Avg. training rmse: %.2f, mae: %.2f, rel_err: %.2f, smape: %.2f' % (epoch, epoch_train_rmse_val, epoch_train_mae_val, epoch_train_rel_err_val, epoch_train_smape_val))

                # Non-regularization
                predicted_y, x_embed_val, epoch_test_rmse_val, epoch_test_mae_val, epoch_test_rel_err_val, epoch_test_smape_val = sess.run([prediction, x_embed, rmse, mae, rel_err, smape], feed_dict={x_ident: test_data_ident, x_remain: test_data_remain, Y: test_label, phase_train: False})

                print ("test: rmse:%.2f" % (epoch_test_rmse_val))
                print ("test: mae: %.2f" % (epoch_test_mae_val))
                print ("test: rel_err: %.2f" % (epoch_test_rel_err_val))
                print ("test: smape: %.2f" % (epoch_test_smape_val))
                print ("(truth, prediction):") 
                print (np.c_[test_label[:10], predicted_y[:10]])

                epoch_list.append (epoch)
                train_rmse_list.append (epoch_train_rmse_val)
                train_mae_list.append (epoch_train_mae_val)
                train_rel_err_list.append (epoch_train_rel_err_val)
                train_smape_list.append (epoch_train_smape_val)
                rmse_list.append (epoch_test_rmse_val)
                mae_list.append (epoch_test_mae_val)
                rel_err_list.append (epoch_test_rel_err_val)
                smape_list.append (epoch_test_smape_val)

                line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
                line['truth'] = test_label.reshape (test_label.shape[0])
                line['pred'] = predicted_y.reshape (predicted_y.shape[0])

                # Save predicted label and determine the best epoch
                if loss_func == "rel_err":
                    threshold_err = 6 #7 #6.5 #9.3 #8.5 #
                    epoch_test_err_val = epoch_test_rel_err_val

                #if (epoch + 1) % 1 == 0:
                if (epoch_test_err_val < threshold_err) or (epoch == nu_epoch - 1):
                    print (epoch_test_err_val, threshold_err)
                    np.savetxt (x_embed_file_name_ + "_" + str (epoch), x_embed_val, fmt="%.2f\t%.2f\t%.2f")
                    np.savetxt (y_predict_file_name_ + "_" + str (epoch), line, fmt="%.2f\t%.2f")
                    save_path = saver.save (sess, model_path + "_" + str (epoch)) #, global_step=global_step)
                    print('Model saved in file: %s' % save_path)

                    ###########
                    ## Test
                    # SAVE THE MODEL INTO .pb formart, recently used only for java
                    builder = tf.saved_model.builder.SavedModelBuilder(model_path + str (epoch))
                    builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING])
                    save_path = builder.save()
                    print('Model saved in file: %s' % save_path)

                if epoch_test_err_val < smallest_epoch_test_err_val:
                    smallest_epoch_test_err_val = epoch_test_err_val 
                    best_epoch = epoch 

                # training data permutation
                train_set_shuffled = np.random.permutation(train_set)
                train_data_remain_shuffled = train_set_shuffled [:, 0:d_remain]
                train_data_ident_shuffled = train_set_shuffled [:, d_remain:d_data]
                train_label_shuffled = train_set_shuffled [:, d_data:d_data+1]
                train_car_ident_shuffled = train_set_shuffled [:, d_data+1:]


            print ("Training finished!")
            stop_time = time.time()
            print ("Training time (s): %.2f" % (stop_time - start_time))
            
            print('test last epoch rel_err: {:.3f}'.format(epoch_test_rel_err_val))

            line = np.zeros(len (epoch_list), dtype=[('epoch', int), ('rmse', float), ('mae', float), ('rel_err', float), ('smape', float), ('train_rmse', float), ('train_mae', float), ('train_rel_err', float), ('train_smape', float)])
            line['epoch'] = epoch_list
            line['rmse'] = rmse_list
            line['mae'] = mae_list
            line['rel_err'] = rel_err_list
            line['smape'] = smape_list
            line['train_rmse'] = train_rmse_list
            line['train_mae'] = train_mae_list
            line['train_rel_err'] = train_rel_err_list
            line['train_smape'] = train_smape_list
            np.savetxt(mean_error_file_name_ + "_" + str (epoch), line, fmt="%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f")

            return best_epoch


    def train_car2vec (self, train_data, train_label, total_car_ident, d_ident, d_embed, d_remain, no_neuron, no_neuron_embed, loss_func, model_path, nu_epoch): 
        #building car embedding model
        x_ident, x_remain, Y, x_embed, prediction, phase_train = self.build_car2vec_model(no_neuron, no_neuron_embed, d_ident, d_embed, d_remain)

        # Try to use weights decay
        num_batches_per_epoch = int (len (train_data) / self.batch_size)
        decay_steps = int (num_batches_per_epoch * self.decay_step)
        global_step = tf.Variable (0, trainable = False)
        learning_rate = tf.train.exponential_decay (self.learning_rate, global_step, decay_steps, self.decay_rate, staircase = True)

        # Used for minimizing relative error
        if loss_func == "mse":
            loss = tf.reduce_mean (tf.pow (prediction - Y, 2)) 
        elif loss_func == "mae":
            loss = tf.reduce_mean (tf.abs (prediction - Y)) #protect when Y = 0
        elif loss_func == "rel_err":
            loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), Y)) #protect when Y = 0
        elif loss_func == "smape":
            loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) ))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Declare error functions
        sum_se = tf.reduce_sum (tf.squared_difference(prediction, Y), name = "sum_se")
        sum_ae = tf.reduce_sum (tf.abs (prediction - Y), name = "sum_ae")
        #sum_rel_err = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y), tf.maximum (Y, prediction))), 100, name = "sum_rel_err")
        #arr_rel_err = tf.multiply (tf.divide (tf.abs (prediction - Y), tf.maximum (Y, prediction)), 100, name = "arr_rel_err")
        sum_rel_err = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y), Y)), 100, name = "sum_rel_err")
        arr_rel_err = tf.multiply (tf.divide (tf.abs (prediction - Y), Y), 100, name = "arr_rel_err")
        sum_smape = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) )), 100, name = "sum_smape")
        rmse = tf.sqrt (tf.reduce_mean(tf.squared_difference(prediction, Y)), name = "rmse")
        mae = tf.reduce_mean (tf.abs (prediction - Y), name = "mae")
        rel_err = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y), (Y))), 100, name = "rel_err") 
        smape = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) )), 100, name = "smape")

        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=100)

            # Run the initializer
            sess.run(init)

            """ Start training, using training data"""
            # record the time when training starts
            start_time = time.time()

            len_train = len(train_data)
            d_data = train_data.shape[1]
            train_car_ident = total_car_ident [:len_train, :]


            train_set = np.concatenate ((train_data, train_label, train_car_ident), axis = 1)
            train_set_shuffled = np.random.permutation(train_set)
            train_data_remain_shuffled = train_set_shuffled [:, 0:d_remain]
            train_data_ident_shuffled = train_set_shuffled [:, d_remain:train_data.shape[1]]
            train_label_shuffled = train_set_shuffled [:, d_data:d_data+1]
            train_car_ident_shuffled = train_set_shuffled [:, d_data+1:]

            print ("len train_data_ident: ", (train_data_ident_shuffled.shape))
            print ("len train_data_remain: ", (train_data_remain_shuffled.shape))
            print ("len train_label: ", (train_label_shuffled.shape))

            len_train = len(train_data)
            train_total_batch = int (np.ceil (float (len_train)/self.batch_size))

            epoch_list = [] 
            train_rel_err_list = [] 
            rmse_list = []
            mae_list = []
            rel_err_list = []
            smape_list = []

            #N = tqdm.trange (nu_epoch)
            #for epoch in N:
            for epoch in tqdm (range (nu_epoch)):
            #for epoch in range (nu_epoch):

                # Train the model.
                total_se = 0
                total_ae = 0
                total_rel_err = 0
                total_smape = 0
                start_index = 0
                end_index = 0
                for i in range (train_total_batch):
                    if len_train - end_index < self.batch_size:
                        end_index = len_train
                    else:
                        end_index = (i+1) * self.batch_size

                    batch_x_ident = train_data_ident_shuffled [start_index : end_index]
                    batch_car_ident = train_car_ident_shuffled [start_index : end_index]
                    batch_x_remain = train_data_remain_shuffled [start_index : end_index]
                    batch_y = train_label_shuffled [start_index : end_index]

                    start_index = end_index

                    _, train_sum_se_val, train_sum_ae_val, train_sum_rel_err_val, train_sum_smape_val = sess.run([optimizer, sum_se, sum_ae, sum_rel_err, sum_smape], feed_dict={x_ident: batch_x_ident, x_remain: batch_x_remain, Y: batch_y, phase_train: True})
                    total_se += train_sum_se_val
                    total_ae += train_sum_ae_val
                    total_rel_err += train_sum_rel_err_val
                    total_smape += train_sum_smape_val

                assert end_index == len_train   

                epoch_train_rmse_val = np.sqrt (total_se/len_train)
                epoch_train_mae_val = total_ae/len_train
                epoch_train_rel_err_val = total_rel_err/len_train
                epoch_train_smape_val = total_smape/len_train
                print('\n\nEpoch: %04d' % (epoch), "Avg. training rmse:", epoch_train_rmse_val, "mae:", epoch_train_mae_val, 'rel_err:', epoch_train_rel_err_val, "smape:", epoch_train_smape_val)
                
                # Save the model
                if epoch == nu_epoch-1 or (epoch + 1)%5 == 0: 
                    ckpt_file = model_path + "_" + str (epoch)
                    save_path = saver.save (sess, ckpt_file)  
                    print('Model saved in file: %s' % save_path)

                # Training data permutation
                train_set_shuffled = np.random.permutation(train_set)
                train_data_remain_shuffled = train_set_shuffled [:, 0:d_remain]
                train_data_ident_shuffled = train_set_shuffled [:, d_remain:d_data]
                train_label_shuffled = train_set_shuffled [:, d_data:d_data+1]
                train_car_ident_shuffled = train_set_shuffled [:, d_data+1:]


    def remove_outliers_total_set (self, total_data, total_label, total_car_ident_code, act_ad_date, sale_date, total_rel_err, removal_percent):
        """
            Purpose:
            - Remove outliers of the total dataset: the data points in the training set with the corresponding relative error in top (removal_percent)%
            - Then, sort the dataset by actual_advertising_date
            Note:
            - If the array total_car_ident_code is empty -> baseline method
            - Else -> car2vec
        """
        print ("Remove outliers from the original array")
        dataset = np.concatenate ((total_data, total_label, total_car_ident_code, act_ad_date, sale_date, total_rel_err), axis=1)

        # Get the dataset after removing removal_percent no. data points with the highest relative error from the total dataset
        len_dataset = total_data.shape[0]
        remain_len = int (len_dataset * (1 - removal_percent / 100.0) + 0.5)

        # Get the indexes of the dataset sorted by rel_err (the last column), (remain_len - 1) data points with rel_err < (remain_len)th data point's rel_err
        new_idx = np.argpartition (dataset[:,-1], remain_len)

        # The indexes of (remain_len) data points with the smallest rel_err.
        idx = new_idx [:remain_len]
        rm_idx = new_idx [remain_len:]
        new_dataset = dataset [idx]

        # Sort by adv_date
        new_dataset = new_dataset [new_dataset[:, -3].argsort()]

        # The new dataset will store sale_date at the end of the matrix
        new_dataset = new_dataset[:, :-1]  

        return (new_dataset, rm_idx)

    def convert_graph_type (self, meta_file, ckpt_file, frozen_graph_file):
        output_node_names = ["prediction"] # Output nodes
      
        with tf.Session () as sess:
            # Restore the graph
            saver = tf.train.import_meta_graph (meta_file)
            # Load weights
            saver.restore (sess, ckpt_file)
            # Freeze the graph
            frozen_graph_def = tf.graph_util.convert_variables_to_constants (sess, sess.graph_def, output_node_names)

            # Save the frozen graph
            with open (frozen_graph_file, "wb") as f:
                f.write (frozen_graph_def.SerializeToString())

    def test_car2vec (self, data, label, d_ident, d_remain, res_file_name, meta_file, ckpt_file):
        print ("Testing...")
        (predicted_label, rmse, mae, avg_rel_err, smape, arr_rel_err) = self.batch_computation_car2vec (5, data, label, d_ident, d_remain, meta_file, ckpt_file, 1)
        np_arr = np.concatenate ((label, predicted_label, arr_rel_err), axis=1)
        print ("Test relative error: %.2f" % (avg_rel_err))
        df = pd.DataFrame (np_arr)
        np.savetxt (res_file_name, df, fmt="%d\t%.0f\t%.2f")
        return 0


    def remove_outliers_total_set_car2vec (self, total_data, total_label, total_car_ident_code, total_act_adv_date, total_sale_date, d_ident, d_remain, y_predict_file_name, removal_percent, epoch):
        """
            - Purpose: 
                + Train the model car2vec with the whole dataset, and then remove the data points with removal_percent highest relative error.
                + Sort the remaining dataset by total_act_adv_date, and divide it into train and test sets
                + Retrain the model car3vect with the new training data (if the "retrain" flag == 1 -> use the initial weights from the 1st train, 
                otherwise retrain from scratch.
        """
        ##########################
        # First train the model on the original train data (can remove a part of outliers previously)
        model_path = self.model_dir + "car2vec_{0}x1_{1}x1".format (self.no_neuron_embed, self.no_neuron)

        print ("\n\n===========Train total set")
        if epoch < 0:
            print ("=== Train from scratch")
            self.train_car2vec (total_data, total_label, total_car_ident_code, d_ident, self.d_embed, d_remain, self.no_neuron, self.no_neuron_embed, self.loss_func, model_path, self.epoch1)
            self.restored_epoch = self.epoch1-1

        meta_file = model_path + "_" + str (self.restored_epoch) + ".meta"
        ckpt_file = model_path + "_" + str (self.restored_epoch)
        print ("=== Restore the trained model from: %s" % (meta_file))

        # Devide the train set into smaller subsets (Eg. 5 subsets), push them to the model and concatenate the predictions later
        (predicted_total_label, total_rmse_val, total_mae_val, total_rel_err_val, total_smape_val, total_arr_rel_err) = self.batch_computation_car2vec (5, total_data, total_label, d_ident, d_remain, meta_file, ckpt_file, 1)
        total_np_arr = np.concatenate ((total_car_ident_code, total_act_adv_date, total_sale_date, total_label, predicted_total_label), axis=1)
        total_df = pd.DataFrame (total_np_arr)
        np.savetxt (y_predict_file_name + "_total_before_remove_outliers", total_df, fmt="%d\t%d\t%d\t%d\t%s\t%.0f\t%d\t%d\t%.2f")

        # Remove outliers from the total dataset based on the relative error from the first train, on the other hand sort the dataset by total_act_adv_date
        stime = time.time()
        if removal_percent > 0:
            new_total_set, rm_idx = self.remove_outliers_total_set (total_data, total_label, total_car_ident_code, total_act_adv_date, total_sale_date, total_arr_rel_err, removal_percent)
            np.savetxt ("./test_rm_idx.txt", rm_idx, fmt="%d")
        else:
            raise ValueError ("Removal perentage is 0!")
        print ("Time for remove outliers from dataset: %.2f" % (time.time() - stime))


    def batch_computation_car2vec (self, no_batch, data, label, d_ident, d_remain, meta_file, ckpt_file, flag):
        data_remain = data [:, 0:d_remain]
        data_ident = data [:, d_remain:]
        # devide the dataset into smaller subsets, push them to the model and concatenate them later
        total_se = 0
        total_ae = 0
        total_rel_err = 0
        total_smape = 0
        start_index = 0
        end_index = 0
        total_predicted_y = np.zeros ((0, 1))
        total_arr_rel_err = np.zeros ((0, 1))
        total_x_embed_val = np.zeros ((0, self.d_embed))
        len_data = len(data)
        batch_size = int (len_data / no_batch)
        total_batch = int (np.ceil (float (len_data)/batch_size))
        with tf.Session () as sess:
            # Load meta graph and restore all variable values
            s_time = time.time ()
            # Restore the graph
            saver = tf.train.import_meta_graph (meta_file)
            # Load weights
            saver.restore (sess, ckpt_file)
            #print ("Time for loading the trained model: %.2f" % (time.time() - s_time))
            
            for i in tqdm (range(total_batch)):
                if len_data - end_index < batch_size:
                    end_index = len_data
                else:
                    end_index = (i+1) * batch_size

                batch_x_ident = data_ident [start_index : end_index]
                batch_x_remain = data_remain [start_index : end_index]
                batch_y = label [start_index : end_index]

                start_index = end_index

                (predicted_y, sum_se_val, sum_ae_val, sum_rel_err_val, sum_smape_val, arr_rel_err_val) = self.restore_model_car2vec (sess, batch_x_ident, batch_x_remain, batch_y, meta_file, ckpt_file, flag)
                total_predicted_y = np.concatenate ((total_predicted_y, predicted_y))
                total_arr_rel_err = np.concatenate ((total_arr_rel_err, arr_rel_err_val))
                total_se += sum_se_val
                total_ae += sum_ae_val
                total_rel_err += sum_rel_err_val
                total_smape += sum_smape_val

            assert end_index == len_data   

            predicted_label = total_predicted_y
            rmse_val = np.sqrt (total_se/len_data)
            mae_val = total_ae/len_data
            avg_rel_err_val = total_rel_err/len_data
            smape_val = total_smape/len_data

            return (predicted_label, rmse_val, mae_val, avg_rel_err_val, smape_val, total_arr_rel_err)
        
    
