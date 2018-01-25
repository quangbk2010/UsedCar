
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:00:00 2017

@author: quang
"""
from Header import *
from Dataset import *
from Performance import *

"""
    ########################
    ### SHORT-LIST MODELS
    ########################
     - Implement, apply algorithms, build models: 
        + Linear, Ridge, Lasso
        + Decision tree: "DecisionTreeRegressor", "GradientBoostingRegressor", "RandomForestRegressor", "AdaBoostRegressor"
        + Deep learning: baseline, car2vect
        + Ensemble methods: 
"""
        
class Sklearn_model (Dataset):
    def __init__(self, dataset_size):
        
        """
            Initialize parameters for Training class
        """
        self.grid_file = "./Model/[" + dataset_size  +  "]grid_search_DT.sav_" 

    def get_predicted_label (self, regr, data):
        return  np.array (regr.predict (data)).reshape (data.shape[0], 1)

    def sklearn_regression (self, reg_type, train_data, train_label, test_data, test_label):
        """
            Use GridSearch to find the best hyper-parameters for each model
        """
        stime = time.time()
        print ("\n\n#########################")
        print ("Model:", reg_type)
        print ("#########################")
        grid_file = self.grid_file
        grid_file += reg_type

        if reg_type == "Linear":
            regr = linear_model.LinearRegression () 
            param_grid = [ {"n_jobs":[-1]}]

        elif reg_type == "Ridge":
            regr = linear_model.Ridge () 
            param_grid = [ {"alpha":[10e-3, 10e-2, 0.5, 1, 2, 5, 10], "max_iter":[50]}]

        elif reg_type == "Lasso":
            regr = linear_model.Lasso () 
            param_grid = [ {"alpha":[10e-3, 10e-2, 0.5, 1, 2, 5, 10], "max_iter":[50]}]

        elif reg_type == "DecisionTreeRegressor":
            regr = tree.DecisionTreeRegressor() 
            param_grid = [ {"criterion":["mae"], "min_samples_split":[5, 10, 20], "max_depth":[10, 20, 30]}]

        elif reg_type == "GradientBoostingRegressor":
            regr = ensemble.GradientBoostingRegressor()
            param_grid = [ {"n_estimators": [10, 100, 500, 1000],"learning_rate": [0.001, 0.01, 0.1, 0.2], "max_depth":[10,20,30]}]

        elif reg_type == "AdaBoostRegressor":
            regr = ensemble.AdaBoostRegressor()
            param_grid = [ {"n_estimators": [10, 100, 500, 1000], "min_samples_split": [5, 10, 20], "max_depth": [10, 20, 30]}]

        elif reg_type == "RandomForestRegressor":
            regr = ensemble.RandomForestRegressor()
            param_grid = [ {"n_estimators": [10, 100, 500, 1000], "min_samples_split": [5, 10, 20], "max_depth": [10, 20, 30]},
                           {"bootstrap": [False], "n_estimators": [10, 100, 500, 1000], "min_samples_split": [5, 10, 20], "max_depth": [10, 20, 30]} 
            ]
        else:
            raise ValueError ("This model is not supported!")

        err = make_scorer (get_relative_err, greater_is_better=False)

        print ("========= Training...")
        if os.path.isfile (grid_file) == False:
            grid_search = GridSearchCV (regr, param_grid, cv = 5, scoring = err) #err, "neg_mean_squared_error"
            grid_search.fit (train_data, train_label)
            pickle.dump (grid_search, open (grid_file, 'wb'))
        else:
            print ("load the gridsearch from disk")
            grid_search = pickle.load (open (grid_file, 'rb'))

        cvres = grid_search.cv_results_
        for mean_score, params in zip (cvres["mean_test_score"], cvres["params"]):
            print (mean_score, params)
        print ("Best params   :", grid_search.best_params_)
        print ("Best estimator:", grid_search.best_estimator_)

        # Testing
        print ("========= Testing...")
        predicted_train_label = self.get_predicted_label (grid_search, train_data) 
        (train_mae, train_rmse, train_rel_err, train_smape) = get_err (predicted_train_label, train_label)
        
        predicted_test_label = self.get_predicted_label (grid_search, test_data) 
        (test_mae, test_rmse, test_rel_err, test_smape) = get_err (predicted_test_label, test_label) 
        print (reg_type + ": (mae, rmse, rel_err, smape)")
        print (" Train err: (%.3f, %.3f, %.3f %%, %.3f %%)" % (train_mae, train_rmse, train_rel_err, train_smape))
        print (" Test err : (%.3f, %.3f, %.3f %%, %.3f %%)" % (test_mae, test_rmse, test_rel_err, test_smape) )

    def __str__(self):
        """
            Display a string representation of the data.
        """
        output = ''
        return (output)

    
class Tensor_NN (Dataset, Sklearn_model):

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

        self.d_embed = args.d_embed
        self.no_hidden_layer = args.no_hidden_layer
        self.no_neuron = args.no_neuron
        self.no_neuron_embed = args.no_neuron_embed
        self.loss_func = args.loss_func
        self.car_ident_flag = args.car_ident_flag

        self.scale_label = args.scale_label
        self.label = args.label
        self.num_regressor = args.num_regressor
        self.sample_ratio = args.sample_ratio
        
        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.args = args

        # These below vars are used in minmax scaler for price
        self.min_price = 1.0
        self.max_price = 100.0

    
    def batch_norm (self, x, phase_train, scope='bn'):
        """
        Batch normalization.
        Args:
            x:           Tensor
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope (scope):
            normed = tf.layers.batch_normalization (x, training=phase_train)
        return normed

    def build_model (self, dim_data, no_unit, no_hidden_layer):
        X = tf.placeholder (tf.float32, [None, dim_data], name="X")
        Y = tf.placeholder (tf.float32, [None, 1], name="Y")

        net = slim.fully_connected (X, no_unit, scope='hidden_layer1', activation_fn=tf.nn.relu) #None) # None) #
        net = slim.dropout (net, self.dropout, scope='dropout1')
        for i in range (1, no_hidden_layer):
            net = slim.fully_connected (net, no_unit, scope='hidden_layer'+str(i+1), activation_fn=tf.nn.relu) #None) # None) #
            net = slim.dropout (net, self.dropout, scope='dropout'+str(i+1))

        prediction = slim.fully_connected (net, 1, scope='output_layer', activation_fn=None) #, reuse=tf.AUTO_REUSE)
        tf.identity (prediction, name="prediction")

        return X, Y, prediction

    def restore_model_NN_baseline (self, data, label, meta_file, ckpt_file):
        with tf.Session() as sess:
            # Load meta graph and restore all variable values
            saver = tf.train.import_meta_graph (meta_file)
            saver.restore (sess, ckpt_file)

            # Access and create placeholder variables, and create feedict to feed data 
            graph = tf.get_default_graph ()
            X = graph.get_tensor_by_name ("X:0")
            Y = graph.get_tensor_by_name ("Y:0")
            feed_dict = {X:data, Y:label}

            # Now, access the operators that we want to run
            prediction = graph.get_tensor_by_name ("prediction:0")
            rmse= graph.get_tensor_by_name ("rmse:0")
            mae = graph.get_tensor_by_name ("mae:0")
            relative_err = graph.get_tensor_by_name ("relative_err:0")
            smape = graph.get_tensor_by_name ("smape:0")
            
            # Feed data
            predicted_y, rmse_val, mae_val, relative_err_val, smape_val = sess.run([prediction, rmse, mae, relative_err, smape], feed_dict)
            return (predicted_y, rmse_val, mae_val, relative_err_val, smape_val)

    
    def baseline (self, train_data, train_label, test_data, test_label, no_neuron, no_hidden_layer, loss_func, model_path, y_predict_file_name, mean_error_file_name): # Used for 1train-1test
    #def baseline (self, train_data, train_label, test_data, test_label, model_path, X, Y, prediction, weights, fold): # used for Cross-validation 
       
        #building car embedding model
        if using_CV_flag == 0:
            X, Y, prediction = self.build_model(train_data.shape[1], no_neuron, no_hidden_layer) 
            mean_error_file_name_ = mean_error_file_name
            y_predict_file_name_ = y_predict_file_name
        else:
            mean_error_file_name_ = mean_error_file_name + "_" + "fold" + str (fold+1)
            y_predict_file_name_ = y_predict_file_name + "_" + "fold" + str (fold+1)


        # Try to use weights decay
        num_batches_per_epoch = int(len(train_data) / self.batch_size)
        decay_steps = int(num_batches_per_epoch * self.decay_step)
        global_step = tf.Variable(0, trainable = False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, decay_steps, self.decay_rate, staircase = True)

        # Used for minimizing relative error
        if loss_func == "mse":
            loss = tf.reduce_mean (tf.pow (prediction - Y, 2)) 
        elif loss_func == "mae":
            loss = tf.reduce_mean (tf.abs (prediction - Y)) 
        elif loss_func == "rel_err":
            loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), Y)) #protect when Y = 0
        elif loss_func == "smape":
            loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) ))

        optimizer = tf.train.AdamOptimizer(learning_rate, name="adam_opt").minimize(loss, global_step=global_step)
    
        # Use in case of scale label
        if self.scale_label == 1:
            train_label_min = train_label.min ()
            train_label_max = train_label.max ()
            print ("train_label_min:", train_label_min, "train_label_max", train_label_max)
            prediction = train_label_min + (prediction - self.min_price) * (train_label_max - train_label_min) / (self.max_price - self.min_price)
            Y_ = train_label_min + (Y - self.min_price) * (train_label_max - train_label_min) / (self.max_price - self.min_price)
 
        else:
            Y_ = Y 

        # Declare error functions
        sum_se = tf.reduce_sum (tf.squared_difference(prediction, Y_))
        sum_ae = tf.reduce_sum (tf.abs (prediction - Y_))
        sum_relative_err = tf.reduce_sum (tf.divide (tf.abs (prediction - Y_), Y_)) * 100 
        sum_smape = tf.reduce_sum (tf.divide (tf.abs (prediction - Y_), tf.abs (Y_) + tf.abs (prediction) )) * 100
        rmse = tf.sqrt (tf.reduce_mean(tf.squared_difference(prediction, Y_)), name = "rmse")
        mae = tf.reduce_mean (tf.abs (prediction - Y_), name = "mae")
        relative_err = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y_), (Y_))), 100, name = "relative_err") 
        smape = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y_), tf.abs (Y_) + tf.abs (prediction) )), 100, name = "smape")
        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=100)

            # Run the initializer
            sess.run(init)

            """ Start training, using training data"""
            # record the time when training starts
            start_time = time.time()

            train_set = np.concatenate ((train_data, train_label), axis = 1)
            train_set_shuffled = np.random.permutation(train_set)
            train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
            train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

            scaler = MinMaxScaler(feature_range=(self.min_price, self.max_price))
            scaler.fit (train_label_shuffled)
            if self.scale_label == 1:
                train_label_shuffled_scaled = scaler.transform (train_label_shuffled)
            else:
                train_label_shuffled_scaled = train_label_shuffled

            len_train = len(train_data)
            len_test  = len(test_data)
            train_total_batch = int (np.ceil (float (len_train)/self.batch_size))
            test_total_batch = int (np.ceil (float (len_test)/self.batch_size))
            smallest_epoch_test_err_val = 10e6
            best_epoch = 0

            epoch_list = [] 
            train_err_list = [] 
            rmse_list = []
            mae_list = []
            rel_err_list = []
            smape_list = []

            for epoch in range (self.epoch):

                # Train the model.
                total_se = 0
                total_ae = 0
                total_relative_err = 0
                total_smape = 0
                start_index = 0
                end_index = 0
                for i in range (train_total_batch):
                    if len_train - end_index < self.batch_size:
                        end_index = len_train
                    else:
                        end_index = (i+1) * self.batch_size

                    batch_x = train_data_shuffled [start_index : end_index]
                    batch_y = train_label_shuffled_scaled [start_index : end_index]

                    start_index = end_index

                    _, train_sum_se_val, train_sum_ae_val, train_sum_relative_err_val, train_sum_smape_val = sess.run([optimizer, sum_se, sum_ae, sum_relative_err, sum_smape], feed_dict={X: batch_x, Y: batch_y})
                    total_se += train_sum_se_val
                    total_ae += train_sum_ae_val
                    total_relative_err += train_sum_relative_err_val
                    total_smape += train_sum_smape_val
                    
                assert end_index == len_train   

                epoch_train_rmse_val = np.sqrt (total_se/len_train)
                epoch_train_mae_val = total_ae/len_train
                epoch_train_relative_err_val = total_relative_err/len_train
                epoch_train_smape_val = total_smape/len_train
                print('\n\nEpoch: %03d' % (epoch), "Avg. training rmse:", epoch_train_rmse_val, "mae:", epoch_train_mae_val, 'relative_err:', epoch_train_relative_err_val, "smape:", epoch_train_smape_val)
                
                # Test the model.
                if self.scale_label == 1:
                    test_label_scaled = scaler.transform (test_label)
                else:
                    test_label_scaled = test_label

                predicted_y, epoch_test_rmse_val, epoch_test_mae_val, epoch_test_relative_err_val, epoch_test_smape_val = sess.run([prediction, rmse, mae, relative_err, smape], feed_dict={X: test_data, Y: test_label_scaled})
                
                print ("test: rmse", epoch_test_rmse_val)
                print ("test: mae", epoch_test_mae_val)
                print ("test: relative_err", epoch_test_relative_err_val)
                print ("test: smape", epoch_test_smape_val)
                #predicted_y_rescaled = train_label_min + (predicted_y - self.min_price) * (train_label_max - train_label_min) / (self.max_price - self.min_price)
                #print ("truth:", test_label[:10], "prediction:", predicted_y_rescaled[:10])
                print ("(truth, prediction):", np.c_[test_label[:10], predicted_y[:10]])

                epoch_list.append (epoch)
                train_err_list.append (epoch_train_relative_err_val)
                rmse_list.append (epoch_test_rmse_val)
                mae_list.append (epoch_test_mae_val)
                rel_err_list.append (epoch_test_relative_err_val)
                smape_list.append (epoch_test_smape_val)

                #line = np.zeros(len (test_label), dtype=[('truth_scaled', float), ('pred_scaled', float), ('truth', float), ('pred', float)])
                line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
                #line['truth_scaled'] = test_label_scaled.reshape (test_label.shape[0])
                #line['pred_scaled'] = predicted_y.reshape (predicted_y.shape[0])
                line['truth'] = test_label.reshape (test_label.shape[0])
                line['pred'] = predicted_y.reshape (predicted_y.shape[0])

                # Save predicted label and determine the best epoch
                if loss_func == "rel_err":
                    threshold_err = 8.5 #9.5# 
                    epoch_test_err_val = epoch_test_relative_err_val
                elif loss_func == "mae":
                    threshold_err = 96
                    epoch_test_err_val = epoch_test_mae_val
                elif loss_func == "mse":
                    threshold_err = 146
                    epoch_test_err_val = epoch_test_rmse_val

                #if (epoch + 1) % 1 == 0:
                if epoch_test_err_val < threshold_err:
                    np.savetxt (y_predict_file_name_ + "_" + str (epoch), line, fmt="%.2f\t%.2f")
                    save_path = saver.save (sess, model_path + "_" + str (epoch)) #, global_step=global_step)
                    print('Model saved in file: %s' % save_path)

                if epoch_test_err_val < smallest_epoch_test_err_val:
                    smallest_epoch_test_err_val = epoch_test_err_val 
                    best_epoch = epoch

                #TODO: training data permutation
                train_set_shuffled = np.random.permutation(train_set)
                train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
                train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]
                if self.scale_label == 1:
                    train_label_shuffled_scaled = scaler.transform (train_label_shuffled)
                else:
                    train_label_shuffled_scaled = train_label_shuffled

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

            return best_epoch

    def build_car2vect_model (self, no_neuron, no_neuron_embed, d_ident, d_embed, d_remain):
        """
            - Args:
                + no_neuron: the number of neuron n 'main' hiddenlayers.
                + no_neuron_embed: the number of neuron in 'embedding' hiddenlayers.
                + d_ident: the dimension of the one-hot vector of a car identification (manufacture_code, ..., rating_code)
                + d_remain: the dimension of the remaining features (after one-hot encoding)
            - Purpose: Create the model of NN with using car2vect: from the one-hot encode of a car identification, use word2vect to embed it into a vector with small dimension.
        """
        x_ident = tf.placeholder(tf.float32, [None, d_ident], name="x_ident")
        x_remain = tf.placeholder(tf.float32, [None, d_remain], name="x_remain")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        print ("build_car2vect_model: d_ident:", d_ident, "d_remain:", d_remain, "d_embed:", d_embed, "no_neuron_embed:", no_neuron_embed, "no_neuron_main:", no_neuron)
        
        # He initializer
        he_init = tf.contrib.layers.variance_scaling_initializer ()
        output1 = slim.fully_connected (x_ident, no_neuron_embed, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=he_init) #None) #
        output1 = slim.dropout (output1, self.dropout, scope='dropout1')
        #output2 = slim.fully_connected (output1, no_neuron_embed, scope='hidden_embed2', activation_fn=tf.nn.relu)
        #output2 = slim.dropout (output2, self.dropout, scope='dropout2')
        x_embed = slim.fully_connected (output1, d_embed, scope='output_embed', activation_fn=None, weights_initializer=he_init) #, activation_fn=tf.nn.relu)#, activation_fn=None) # 3-dimension of embeding NN
        #x_embed = slim.fully_connected (output1, d_embed, scope='output_embed', activation_fn=tf.nn.relu) # 3-dimension of embeding NN
        #x_embed = slim.fully_connected (output1, d_embed, scope='output_embed') # seperate the activation function to another step to use batch normalization.
        #x_embed = self.batch_norm (x_embed, phase_train) # batch normalization
        #x_embed = tf.nn.elu (x_embed, name="elu_output_embed")

        #mean, var = tf.nn.moments (x_embed, [0], keep_dims=True)
        #x_embed = tf.div(tf.subtract(x_embed, mean), tf.sqrt(var))
        #x_embed = tf.div (x_embed - tf.reduce_min (x_embed), tf.reduce_max (x_embed) - tf.reduce_min (x_embed))

        input3 = tf.concat ([x_remain, x_embed], 1)

        output3 = slim.fully_connected(input3, no_neuron, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=he_init)
        output3 = slim.dropout (output3, self.dropout, scope='dropout3')
        #output4 = slim.fully_connected(output3, no_neuron, scope='hidden_main_2', activation_fn=tf.nn.relu)
        #output5 = slim.fully_connected(output4, no_neuron, scope='hidden_main_3', activation_fn=tf.nn.relu)
        prediction = slim.fully_connected(output3, 1, scope='output_main')#, activation_fn=tf.nn.relu) #None) # 1-dimension of output NOTE: only remove relu activation function in the last layer if using Gradient Boosting, because the differece can be negative
        tf.identity (prediction, name="prediction")

        return x_ident, x_remain, Y, x_embed, prediction, phase_train

    def build_car2vect_model_retrained (self, no_neuron, no_neuron_embed, d_ident, d_embed, d_remain, train_data, train_label, meta_file, ckpt_file):
        """
            Purpose: Use weights and biases from pre-trained model as initializations
        """
        train_data_remain = train_data [:, 0:d_remain]
        train_data_ident = train_data [:, d_remain:]
        (init_w_hid_embed_1_val, init_bias_hid_embed_1_val, init_w_out_embed_val, init_bias_out_embed_val, init_w_hid_main_1_val, init_bias_hid_main_1_val, init_w_out_main_val, init_bias_out_main_val) = self.restore_weights_car2vect (train_data_ident, train_data_remain, train_label, meta_file, ckpt_file) # TODO: ???why if I leave this line after placeholder initialization, it causes the error: some nodes: x_ident_1, x_remain_1, Y_1, phase_train_1 will be created in the graph?


        x_ident = tf.placeholder(tf.float32, [None, d_ident], name="x_ident")
        x_remain = tf.placeholder(tf.float32, [None, d_remain], name="x_remain")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        print ("build_car2vect_model: d_ident:", d_ident, "d_remain:", d_remain, "d_embed:", d_embed, "no_neuron_embed:", no_neuron_embed, "no_neuron_main:", no_neuron)
        
        output1 = slim.fully_connected (x_ident, no_neuron_embed, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=tf.constant_initializer (init_w_hid_embed_1_val), biases_initializer=tf.constant_initializer (init_bias_hid_embed_1_val)) #None) #
        output1 = slim.dropout (output1, self.dropout, scope='dropout1')
        x_embed = slim.fully_connected (output1, d_embed, scope='output_embed', activation_fn=None, weights_initializer=tf.constant_initializer (init_w_out_embed_val), biases_initializer=tf.constant_initializer (init_bias_out_embed_val)) 

        input3 = tf.concat ([x_remain, x_embed], 1)

        output3 = slim.fully_connected(input3, no_neuron, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=tf.constant_initializer (init_w_hid_main_1_val), biases_initializer=tf.constant_initializer (init_bias_hid_main_1_val))
        output3 = slim.dropout (output3, self.dropout, scope='dropout3')
        prediction = slim.fully_connected(output3, 1, scope='output_main', activation_fn=tf.nn.relu, weights_initializer=tf.constant_initializer (init_w_out_main_val), biases_initializer=tf.constant_initializer (init_bias_out_main_val)) #None) # 1-dimension of output
        tf.identity (prediction, name="prediction")

        return x_ident, x_remain, Y, x_embed, prediction, phase_train
    

    def restore_model_car2vect (self, x_ident_, x_remain_, label, meta_file, ckpt_file, train_flag):
        with tf.Session() as sess:
            # Load meta graph and restore all variable values
            saver = tf.train.import_meta_graph (meta_file)
            saver.restore (sess, ckpt_file)

            # Access and create placeholder variables, and create feedict to feed data 
            graph = tf.get_default_graph ()
            x_ident = graph.get_tensor_by_name ("x_ident:0")
            x_remain = graph.get_tensor_by_name ("x_remain:0")
            Y = graph.get_tensor_by_name ("Y:0")
            feed_dict = {x_ident:x_ident_, x_remain:x_remain_, Y:label}

            # Now, access the operators that we want to run
            prediction = graph.get_tensor_by_name ("prediction:0")

            # And feed data
            if (train_flag == 0):
                rmse= graph.get_tensor_by_name ("rmse:0")
                mae = graph.get_tensor_by_name ("mae:0")
                relative_err = graph.get_tensor_by_name ("relative_err:0")
                smape = graph.get_tensor_by_name ("smape:0")

                predicted_y, rmse_val, mae_val, relative_err_val, smape_val = sess.run([prediction, rmse, mae, relative_err, smape], feed_dict)
                return (predicted_y, rmse_val, mae_val, relative_err_val, smape_val)
            else:
                sum_se= graph.get_tensor_by_name ("sum_se:0")
                sum_ae = graph.get_tensor_by_name ("sum_ae:0")
                sum_rel_err = graph.get_tensor_by_name ("sum_rel_err:0")
                arr_rel_err = graph.get_tensor_by_name ("arr_rel_err:0")
                sum_smape = graph.get_tensor_by_name ("sum_smape:0")

                predicted_y, sum_se_val, sum_ae_val, sum_rel_err_val, sum_smape_val , arr_rel_err_val = sess.run([prediction, sum_se, sum_ae, sum_rel_err, sum_smape, arr_rel_err], feed_dict)
                return (predicted_y, sum_se_val, sum_ae_val, sum_rel_err_val, sum_smape_val, arr_rel_err_val)


    def restore_weights_car2vect (self, x_ident_, x_remain_, label, meta_file, ckpt_file):
        with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run (init)
            # Load meta graph and restore all variables values
            saver = tf.train.import_meta_graph (meta_file)
            saver.restore (sess, ckpt_file)

            # Access and create placeholder variables, and create feedict to feed data
            graph = tf.get_default_graph ()
            x_ident = graph.get_tensor_by_name ("x_ident:0")
            x_remain = graph.get_tensor_by_name ("x_remain:0")
            Y = graph.get_tensor_by_name ("Y:0")
            feed_dict = {x_ident:x_ident_, x_remain:x_remain_, Y:label}

            # Now, access the weights of the pre-trained model
            w_hid_embed_1 = graph.get_tensor_by_name ("hidden_embed1/weights:0")
            bias_hid_embed_1 = graph.get_tensor_by_name ("hidden_embed1/biases:0")

            w_out_embed = graph.get_tensor_by_name ("output_embed/weights:0")
            bias_out_embed = graph.get_tensor_by_name ("output_embed/biases:0")

            w_hid_main_1 = graph.get_tensor_by_name ("hidden_main1/weights:0")
            bias_hid_main_1 = graph.get_tensor_by_name ("hidden_main1/biases:0")

            w_out_main = graph.get_tensor_by_name ("output_main/weights:0")
            bias_out_main = graph.get_tensor_by_name ("output_main/biases:0")

            # Feed data
            #(w_hid_embed_1_val, bias_hid_embed_1_val, w_out_embed_val, bias_out_embed_val, w_hid_main_1_val, bias_hid_main_1_val, w_out_main_val, bias_out_main_val) 
            return sess.run ([w_hid_embed_1, bias_hid_embed_1, w_out_embed, bias_out_embed, w_hid_main_1, bias_hid_main_1, w_out_main, bias_out_main], feed_dict)

    

    def car2vect(self, train_data, train_label, test_data, test_label, test_car_ident, d_ident, d_embed, d_remain, no_neuron, no_neuron_embed, loss_func, model_path, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, retrain): # Used for 1train-1test
    #def car2vect(self, train_data, train_label, test_data, test_label, test_car_ident, model_path, d_ident, d_embed, d_remain, x_ident, x_remain, Y, x_embed, prediction, fold): # used for Cross-validation 

        #building car embedding model
        if using_CV_flag == 0:
            if retrain == 0:
                x_ident, x_remain, Y, x_embed, prediction, phase_train = self.build_car2vect_model(no_neuron, no_neuron_embed, d_ident, d_embed, d_remain)

            ###########
            else:
                self.learning_rate /= 10
                print ("=======new learning_rate:", self.learning_rate)
                pre_model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect/regressor1/full_{0}_{1}_car2vect_{2}_{3}_total_set".format (self.model_name, self.label, self.no_neuron_embed, self.no_neuron)
                meta_file = pre_model_path + ".meta"
                ckpt_file = pre_model_path 
                x_ident, x_remain, Y, x_embed, prediction, phase_train = self.build_car2vect_model_retrained (no_neuron=no_neuron, no_neuron_embed=no_neuron_embed, d_ident=d_ident, d_embed=d_embed, d_remain=d_remain, train_data=train_data, train_label=train_label, meta_file=meta_file, ckpt_file=ckpt_file)
            ###########

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
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, decay_steps, self.decay_rate, staircase = True)

        # Used for minimizing relative error
        if loss_func == "mse":
            loss = tf.reduce_mean (tf.pow (prediction - Y, 2)) 
        elif loss_func == "mae":
            loss = tf.reduce_mean (tf.abs (prediction - Y)) #protect when Y = 0
        elif loss_func == "rel_err":
            loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), Y)) #protect when Y = 0
        elif loss_func == "smape":
            loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) ))

        #lamb = 1e-6
        #loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), Y)) #+ lamb * tf.reduce_mean (tf.norm (x_embed, axis=0, keep_dims=True))
        #loss = tf.reduce_mean (tf.abs (prediction - Y)) #+ lamb * tf.reduce_mean (tf.norm (x_embed, axis=0, keep_dims=True))
        #loss = tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) ))

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
        # Use in case of scale label
        if self.scale_label == 1:
            train_label_min = train_label.min ()
            train_label_max = train_label.max ()
            print ("train_label_min:", train_label_min, "train_label_max", train_label_max)
            prediction = train_label_min + (prediction - self.min_price) * (train_label_max - train_label_min) / (self.max_price - self.min_price)
            Y_ = train_label_min + (Y - self.min_price) * (train_label_max - train_label_min) / (self.max_price - self.min_price)
 
        else:
           Y_ = Y 

        # Declare error functions
        sum_se = tf.reduce_sum (tf.squared_difference(prediction, Y_), name = "sum_se")
        sum_ae = tf.reduce_sum (tf.abs (prediction - Y_), name = "sum_ae")
        sum_relative_err = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y_), Y_)), 100, name = "sum_rel_err")
        arr_relative_err = tf.multiply (tf.divide (tf.abs (prediction - Y_), Y_), 100, name = "arr_rel_err")
        sum_smape = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y_), tf.abs (Y_) + tf.abs (prediction) )), 100, name = "sum_smape")
        rmse = tf.sqrt (tf.reduce_mean(tf.squared_difference(prediction, Y_)), name = "rmse")
        mae = tf.reduce_mean (tf.abs (prediction - Y_), name = "mae")
        relative_err = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y_), (Y_))), 100, name = "relative_err") 
        #median_rel_err = tf.contrib.distributions.percentile (tf.divide (tf.abs (prediction - Y_), (Y_)), 50.0, name = "median_rel_err")
        smape = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y_), tf.abs (Y_) + tf.abs (prediction) )), 100, name = "smape")

        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=100)

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

            if self.scale_label == 1:
                scaler = MinMaxScaler(feature_range=(self.min_price, self.max_price))
                scaler.fit (train_label_shuffled)
                train_label_shuffled_scaled = scaler.transform (train_label_shuffled)
            else:
                train_label_shuffled_scaled = train_label_shuffled

            test_data_remain = test_data [:, 0:d_remain]
            test_data_ident = test_data [:, d_remain:]

            print ("test car ident", test_car_ident[0])
            np.savetxt (x_ident_file_name_, test_car_ident, fmt="%d\t%d\t%d\t%d\t%s")  

            print ("len train_data_ident:", train_data_ident_shuffled.shape)
            print ("len train_data_remain:", train_data_remain_shuffled.shape)
            print ("len train_label:", train_label_shuffled.shape)

            len_train = len(train_data)
            len_test  = len(test_data)
            train_total_batch = int (np.ceil (float (len_train)/self.batch_size))
            test_total_batch = int (np.ceil (float (len_test)/self.batch_size))
            smallest_epoch_test_err_val = 10e6

            epoch_list = [] 
            train_err_list = [] 
            rmse_list = []
            mae_list = []
            rel_err_list = []
            smape_list = []

            for epoch in range (self.epoch):

                # Train the model.
                total_se = 0
                total_ae = 0
                total_relative_err = 0
                total_smape = 0
                start_index = 0
                end_index = 0
                for i in range (train_total_batch):
                    if len_train - end_index < self.batch_size:
                        end_index = len_train
                    else:
                        end_index = (i+1) * self.batch_size

                    batch_x_ident = train_data_ident_shuffled [start_index : end_index]
                    batch_x_remain = train_data_remain_shuffled [start_index : end_index]
                    batch_y = train_label_shuffled_scaled [start_index : end_index]

                    start_index = end_index

                    _, train_sum_se_val, train_sum_ae_val, train_sum_relative_err_val, train_sum_smape_val = sess.run([optimizer, sum_se, sum_ae, sum_relative_err, sum_smape], feed_dict={x_ident: batch_x_ident, x_remain: batch_x_remain, Y: batch_y, phase_train: True})
                    total_se += train_sum_se_val
                    total_ae += train_sum_ae_val
                    total_relative_err += train_sum_relative_err_val
                    total_smape += train_sum_smape_val

                assert end_index == len_train   

                epoch_train_rmse_val = np.sqrt (total_se/len_train)
                epoch_train_mae_val = total_ae/len_train
                epoch_train_relative_err_val = total_relative_err/len_train
                epoch_train_smape_val = total_smape/len_train
                print('\n\nEpoch: %04d' % (epoch), "Avg. training rmse:", epoch_train_rmse_val, "mae:", epoch_train_mae_val, 'relative_err:', epoch_train_relative_err_val, "smape:", epoch_train_smape_val)

                # Test the model.
                if self.scale_label == 1:
                    test_label_scaled = scaler.transform (test_label)
                else:
                    test_label_scaled = test_label

                """# If we use 2 hidden layers, each has >= 10000 units -> resource exhausted, then we should divide it into batches and test on seperate one, and then calculate the average.
                total_se = 0
                total_ae = 0
                total_relative_err = 0
                total_smape = 0
                start_index = 0
                end_index = 0
                total_predicted_y = np.zeros ((0, 1))
                total_x_embed_val = np.zeros ((0, d_embed))

                for i in range (test_total_batch):
                    if len_test - end_index < self.batch_size:
                        end_index = len_test
                    else:
                        end_index = (i+1) * self.batch_size
                    
                    batch_x_ident = test_data_ident [start_index : end_index]
                    batch_x_remain = test_data_remain [start_index : end_index]
                    batch_y = test_label [start_index : end_index]

                    start_index = end_index

                    predicted_y, x_embed_val, test_sum_se_val, test_sum_ae_val, test_sum_relative_err_val, test_sum_smape_val = sess.run([prediction, x_embed, sum_se, sum_ae, sum_relative_err, sum_smape], feed_dict={x_ident: batch_x_ident, x_remain: batch_x_remain, Y: batch_y})

                    total_predicted_y = np.concatenate ((total_predicted_y, predicted_y))
                    total_x_embed_val = np.concatenate ((total_x_embed_val, x_embed_val))
                    total_se += test_sum_se_val
                    total_ae += test_sum_ae_val
                    total_relative_err += test_sum_relative_err_val
                    total_smape += test_sum_smape_val

                assert end_index == len_test   

                predicted_y = total_predicted_y
                x_embed_val = total_x_embed_val
                epoch_test_rmse_val = np.sqrt (total_se/len_test)
                epoch_test_mae_val = total_ae/len_test
                epoch_test_relative_err_val = total_relative_err/len_test
                epoch_test_smape_val = total_smape/len_test"""
                predicted_y, x_embed_val, epoch_test_rmse_val, epoch_test_mae_val, epoch_test_relative_err_val, epoch_test_smape_val = sess.run([prediction, x_embed, rmse, mae, relative_err, smape], feed_dict={x_ident: test_data_ident, x_remain: test_data_remain, Y: test_label_scaled, phase_train: False})

                
                print ("test: rmse", epoch_test_rmse_val)
                print ("test: mae", epoch_test_mae_val)
                print ("test: relative_err", epoch_test_relative_err_val)
                print ("test: smape", epoch_test_smape_val)
                print ("(truth, prediction):", np.c_[test_label[:10], predicted_y[:10]])

                epoch_list.append (epoch)
                train_err_list.append (epoch_train_relative_err_val)
                rmse_list.append (epoch_test_rmse_val)
                mae_list.append (epoch_test_mae_val)
                rel_err_list.append (epoch_test_relative_err_val)
                smape_list.append (epoch_test_smape_val)

                line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
                line['truth'] = test_label.reshape (test_label.shape[0])
                line['pred'] = predicted_y.reshape (predicted_y.shape[0])

                # Save predicted label and determine the best epoch
                if loss_func == "rel_err":
                    threshold_err = 7#6.5 #9.3 #8.5 #
                    epoch_test_err_val = epoch_test_relative_err_val
                elif loss_func == "mae":
                    threshold_err = 150
                    epoch_test_err_val = epoch_test_mae_val
                elif loss_func == "mse":
                    threshold_err = 170 #140
                    epoch_test_err_val = epoch_test_rmse_val

                #if (epoch + 1) % 1 == 0:
                if epoch_test_err_val < threshold_err:
                    print (epoch_test_err_val, threshold_err)
                    np.savetxt (x_embed_file_name_ + "_" + str (epoch), x_embed_val, fmt="%.2f\t%.2f\t%.2f")
                    np.savetxt (y_predict_file_name_ + "_" + str (epoch), line, fmt="%.2f\t%.2f")
                    save_path = saver.save (sess, model_path + "_" + str (epoch)) #, global_step=global_step)
                    print('Model saved in file: %s' % save_path)

                if epoch_test_err_val < smallest_epoch_test_err_val:
                    smallest_epoch_test_err_val = epoch_test_err_val 
                    best_epoch = epoch 

                #TODO: training data permutation
                train_set_shuffled = np.random.permutation(train_set)
                train_data_remain_shuffled = train_set_shuffled [:, 0:d_remain]
                train_data_ident_shuffled = train_set_shuffled [:, d_remain:train_data.shape[1]]
                train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]
                if self.scale_label == 1:
                    train_label_shuffled_scaled = scaler.transform (train_label_shuffled)
                else:
                    train_label_shuffled_scaled = train_label_shuffled


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

            return best_epoch


    def train_car2vect (self, train_data, train_label, d_ident, d_embed, d_remain, no_neuron, no_neuron_embed, loss_func, model_path): 
        #building car embedding model
        x_ident, x_remain, Y, x_embed, prediction, phase_train = self.build_car2vect_model(no_neuron, no_neuron_embed, d_ident, d_embed, d_remain)

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
        sum_relative_err = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y), Y)), 100, name = "sum_rel_err")
        arr_relative_err = tf.multiply (tf.divide (tf.abs (prediction - Y), Y), 100, name = "arr_rel_err")
        sum_smape = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) )), 100, name = "sum_smape")
        rmse = tf.sqrt (tf.reduce_mean(tf.squared_difference(prediction, Y)), name = "rmse")
        mae = tf.reduce_mean (tf.abs (prediction - Y), name = "mae")
        relative_err = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y), (Y))), 100, name = "relative_err") 
        smape = tf.multiply (tf.reduce_mean (tf.divide (tf.abs (prediction - Y), tf.abs (Y) + tf.abs (prediction) )), 100, name = "smape")

        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=100)

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

            print ("len train_data_ident:", train_data_ident_shuffled.shape)
            print ("len train_data_remain:", train_data_remain_shuffled.shape)
            print ("len train_label:", train_label_shuffled.shape)

            len_train = len(train_data)
            train_total_batch = int (np.ceil (float (len_train)/self.batch_size))

            epoch_list = [] 
            train_err_list = [] 
            rmse_list = []
            mae_list = []
            rel_err_list = []
            smape_list = []

            for epoch in range (self.epoch):

                # Train the model.
                total_se = 0
                total_ae = 0
                total_relative_err = 0
                total_smape = 0
                start_index = 0
                end_index = 0
                for i in range (train_total_batch):
                    if len_train - end_index < self.batch_size:
                        end_index = len_train
                    else:
                        end_index = (i+1) * self.batch_size

                    batch_x_ident = train_data_ident_shuffled [start_index : end_index]
                    batch_x_remain = train_data_remain_shuffled [start_index : end_index]
                    batch_y = train_label_shuffled [start_index : end_index]

                    start_index = end_index

                    _, train_sum_se_val, train_sum_ae_val, train_sum_relative_err_val, train_sum_smape_val = sess.run([optimizer, sum_se, sum_ae, sum_relative_err, sum_smape], feed_dict={x_ident: batch_x_ident, x_remain: batch_x_remain, Y: batch_y, phase_train: True})
                    total_se += train_sum_se_val
                    total_ae += train_sum_ae_val
                    total_relative_err += train_sum_relative_err_val
                    total_smape += train_sum_smape_val

                assert end_index == len_train   

                epoch_train_rmse_val = np.sqrt (total_se/len_train)
                epoch_train_mae_val = total_ae/len_train
                epoch_train_relative_err_val = total_relative_err/len_train
                epoch_train_smape_val = total_smape/len_train
                print('\n\nEpoch: %04d' % (epoch), "Avg. training rmse:", epoch_train_rmse_val, "mae:", epoch_train_mae_val, 'relative_err:', epoch_train_relative_err_val, "smape:", epoch_train_smape_val)

            # Save the model
            save_path = saver.save (sess, model_path)
            print('Model saved in file: %s' % save_path)


    def cross_validation (self):
        
        with tf.Session() as sess:
            """ In each iteration, we need to store training error and test error into 2 separare lists"""
            print ("Start k-fold CV:")
            test_err = []
            for i in range (self.k_fold):
                print ("Start fold:", i)
                tf.reset_default_graph ()

                # Prepare data 
                train_data = self.X_train_set[i] 
                train_label = self.y_train_set[i]
                test_data = self.X_test_set[i]
                test_label = self.y_test_set[i]
                if using_car_ident_flag == 1:
                    test_car_ident = self.car_ident_code_test_set[i]

                # Shuffle train data
                train_set = np.concatenate ((train_data, train_label), axis = 1)
                train_set_shuffled = np.random.permutation(train_set)
                train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
                train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

                # Train data corresponding to the type of model
                if using_car_ident_flag == 1:
                    best_epoch = self.car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, test_car_ident=test_car_ident, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func="rel_err", model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_, x_ident_file_name=x_ident_file_name_, x_embed_file_name=x_embed_file_name_, retrain=0)
                else:
                    best_epoch = self.baseline (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, no_neuron=no_neuron, no_hidden_layer=no_hidden_layer, loss_func="rel_err", model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_)

                test_err.append (test_relative_err_val) 
                print ("Fold:", i, test_relative_err_val)
            print ("---Mean relative_err:", np.mean (test_err))
            return np.mean (test_err)
    
    def gradient_boosting_NN_baseline (self, train_data, train_label, test_data, test_label, y_predict_file_name, mean_error_file_name, dataset_size):
        """
            - Purpose: Apply Gradient Boosting (a kind of ensemble method) using baseline NN.
        """
        train_label_copy = train_label.astype (np.float64)
        test_label_copy = test_label.astype (np.float64)
        list_predicted_test_label = []

        for i in range (self.num_regressor):
            print ("\n\n==============regressor%d" %(i+1))
            if i == 0:
                loss_func = "rel_err"
            else:
                loss_func = self.loss_func
            model_path = self.model_dir + "/gb_NN/baseline/regressor" + str (i+1) + "/" + dataset_size + "_" + self.model_name  + "_" + self.label  + "_baseline_" + str (self.no_neuron) + "_" + str (self.no_hidden_layer) + "_" + loss_func
            y_predict_file_name_ = y_predict_file_name + "_" + str (i+1)
            mean_error_file_name_ = mean_error_file_name + "_" + str (i+1)

            if i > 0:
                train_label_copy -= predicted_train_label  
                test_label_copy -= predicted_test_label 
                #train_label_copy = predicted_train_label - train_label_copy
                #test_label_copy = predicted_test_label - test_label_copy

            tf.reset_default_graph ()
            os.system ("mkdir -p ../checkpoint/gb_NN/car2vect/regressor" + str (i+1))
            #best_epoch = self.baseline (train_data=train_data, train_label=train_label_copy, test_data=test_data, test_label=test_label_copy, no_neuron=no_neuron, no_hidden_layer = no_hidden_layer, loss_func=loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_)
            if i == 0:
                best_epoch = 9
                model_path = self.model_dir + "/gb_NN/baseline/regressor" + str (i+1) + "/" + dataset_size + "_" + self.model_name  + "_" + self.label  + "_baseline_2000_2_" + loss_func
            elif i == 1: #i > 0: # 
                best_epoch = 17
                loss_func = "mse"
                model_path = self.model_dir + "/gb_NN/baseline/regressor" + str (i+1) + "/" + dataset_size + "_" + self.model_name  + "_" + self.label  + "_baseline_1000_1_" + loss_func #"_baseline_" + str (self.no_neuron) + "_" + str (self.no_hidden_layer) + "_" + loss_func
            elif i == 2:
                best_epoch = 16
                loss_func = "mse"
                model_path = self.model_dir + "/gb_NN/baseline/regressor" + str (i+1) + "/" + dataset_size + "_" + self.model_name  + "_" + self.label  + "_baseline_1000_1_" + loss_func #"_baseline_" + str (self.no_neuron) + "_" + str (self.no_hidden_layer) + "_" + loss_func
            else:
            #elif i == 2:
                loss_func = "mae"
                best_epoch = self.baseline (train_data=train_data, train_label=train_label_copy, test_data=test_data, test_label=test_label_copy, no_neuron=self.no_neuron, no_hidden_layer=self.no_hidden_layer, loss_func=loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_)
                bash_cmd = "cd ../checkpoint/gb_NN/baseline/regressor" + str (i+1) + "; mkdir temp_save; rm temp_save/*; cp checkpoint *" + str (best_epoch) + "*" + " temp_save; cd ../../../../Code"
                print ("bash_cmd:", bash_cmd)
                os.system (bash_cmd)
                #best_epoch = 0
            print ("Best epoch: ", best_epoch)

            meta_file = model_path + "_" + str (best_epoch) + ".meta"
            ckpt_file = model_path + "_" + str (best_epoch) 
            (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val) = self.restore_model_NN_baseline (train_data, train_label_copy, meta_file, ckpt_file)
            (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_NN_baseline (test_data, test_label_copy, meta_file, ckpt_file)
            print (predicted_test_label[:10], test_label_copy[:10])
            print ("=================================")

            list_predicted_test_label.append (predicted_test_label)

        print ("label", test_label[:10])
        for i in range (self.num_regressor):
            predicted_test_label = sum (pred_y for pred_y in list_predicted_test_label[:i+1])
            print ("predicted_test_label (" + str(i) + ")", predicted_test_label[:10])
            (test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = get_err (predicted_test_label, test_label)
            print ("Err after " + str (i+1)  + " regressors:", test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val)
            line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
            line['truth'] = test_label.reshape (test_label.shape[0])
            line['pred'] = predicted_test_label.reshape (predicted_test_label.shape[0])
            np.savetxt (y_predict_file_name_ + "_final" + str(i), line, fmt="%.2f\t%.2f")

    def gradient_boosting_NN_car2vect (self, train_data, train_label, test_data, test_label, test_car_ident, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, retrain):
        """
            - Purpose: Apply Gradient Boosting (a kind of ensemble method) using car2vect NN.
        """
        train_data_remain = train_data [:, 0:d_remain]
        train_data_ident = train_data [:, d_remain:]
        test_data_remain = test_data [:, 0:d_remain]
        test_data_ident = test_data [:, d_remain:]

        train_label_copy = train_label.astype (np.float64)
        test_label_copy = test_label.astype (np.float64)
        list_predicted_test_label = []

        for i in range (self.num_regressor):
            print ("\n\n==============regressor%d" %(i+1))
            if i == 0:
                loss_func = "rel_err"
                if retrain == 2: # In case of retrain model after removing outliers from the 1st train, retrain_flag = 2, but it should be 1 in the 1st regressor if using Gradient Boosting.
                    retrain_ = 1
            else:
                loss_func = self.loss_func
                if retrain == 2: # In case of retrain model after removing outliers from the 1st train, retrain_flag = 2, but it should be 0 after the 1st regressor if using Gradient Boosting.
                    retrain_ = 0
            model_path = self.model_dir + "/gb_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}_{5}_{6}".format (i+1, dataset_size, self.model_name, self.label, self.no_neuron_embed, self.no_neuron, loss_func)
            y_predict_file_name_ = y_predict_file_name + "_" + str (i+1)
            mean_error_file_name_ = mean_error_file_name + "_" + str (i+1)
            x_ident_file_name_ = x_ident_file_name + "_" + str (i+1)
            x_embed_file_name_ = x_embed_file_name + "_" + str (i+1)

            if i > 0:
                train_label_copy -= predicted_train_label  
                test_label_copy -= predicted_test_label 
                #train_label_copy = predicted_train_label - train_label_copy
                #test_label_copy = predicted_test_label - test_label_copy

            tf.reset_default_graph ()
            os.system ("mkdir -p ../checkpoint/gb_NN/car2vect/regressor" + str (i+1))
            best_epoch = self.car2vect (train_data=train_data, train_label=train_label_copy, test_data=test_data, test_label=test_label_copy, test_car_ident=test_car_ident, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func=loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_, x_ident_file_name=x_ident_file_name_, x_embed_file_name=x_embed_file_name_, retrain=retrain_)
            
            print ("Best epoch: ", best_epoch)
            bash_cmd = "cd ../checkpoint/gb_NN/car2vect/regressor" + str (i+1) + "; mkdir temp_save; rm temp_save/*; cp checkpoint *" + str (best_epoch) + "*" + " temp_save; cd ../../../../Code"
            print ("bash_cmd:", bash_cmd)
            os.system (bash_cmd)

            meta_file = model_path + "_" + str (best_epoch) + ".meta"
            ckpt_file = model_path + "_" + str (best_epoch) 
            # When restore model with the whole dataset, it can cause the error: Resource exhausted 
            # TODO: devide the train set into smaller subsets, push them to the model and concatenate them later
            (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val, total_arr_relative_err) = self.batch_computation_car2vect (5, train_data, train_label_copy, d_ident, d_remain, meta_file, ckpt_file)

            # TODO: when restore model with train data is imported -> it will return sum_se, sum_ae, ... -> need to get the average or sqrt
            # If the length of train data is small as the number of train data in case of testing only Hyundai, Kia -> can use train_flag=0
            #(predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val) = self.restore_model_car2vect (train_data_ident, train_data_remain, train_label_copy, meta_file, ckpt_file, train_flag=1)
            (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_car2vect (test_data_ident, test_data_remain, test_label_copy, meta_file, ckpt_file, train_flag=0)
            print ("(truth, prediction):", np.c_[test_label_copy[:10], predicted_test_label[:10]])
            print ("=================================")

            list_predicted_test_label.append (predicted_test_label)

        print ("label", test_label[:10])
        for i in range (self.num_regressor):
            predicted_test_label = sum (pred_y for pred_y in list_predicted_test_label[:i+1])
            print ("predicted_test_label (" + str(i) + ")", predicted_test_label[:10])
            (test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = get_err (predicted_test_label, test_label)
            print ("Err after " + str (i+1)  + " regressors:", test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val)
            line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
            line['truth'] = test_label.reshape (test_label.shape[0])
            line['pred'] = predicted_test_label.reshape (predicted_test_label.shape[0])
            np.savetxt (y_predict_file_name_ + "_final" + str(i), line, fmt="%.2f\t%.2f")
        
    def remove_outliers (self, train_data, train_label, train_rel_err, removal_percent):
        """
            Purpose: Remove outliers of the dataset: the data points in the training set with the corresponding relative error in top (removal_percent)%
        """
        dataset = np.concatenate ((train_data, train_label, train_rel_err), axis=1)
        len_train = train_data.shape[0]
        remain_len = int (len_train * (1 - removal_percent /100.0)+0.5)
        idx = np.argpartition (dataset[:,-1], remain_len)[:remain_len]
        new_dataset = dataset [idx]
        return new_dataset[:, :-1]

    def remove_outliers_total_set (self, total_data, total_label, total_car_ident_code, act_ad_date, total_rel_err, dataset_size, removal_percent):
        """
            Purpose:
            - Remove outliers of the total dataset: the data points in the training set with the corresponding relative error in top (removal_percent)%
            - Then, sort the dataset by actual_advertising_date
        """
        # Save and load file take a lot of times, much more than the common way
        #stored_np_arr_file = "./Dataframe/[{0}]total_numpy_array_after_remove_outliers.h5_{1}".format (dataset_size, removal_percent)
        #removed_np_arr_file = "./Dataframe/[{0}]removed_numpy_array_after_remove_outliers.h5_{1}".format (dataset_size, removal_percent)
        #stored_np_arr_file = "./Dataframe/[{0}]total_numpy_array_after_remove_outliers.h5_{1}".format (dataset_size, removal_percent)
        #key = "df"
        #if os.path.isfile (stored_np_arr_file + ".npy") == False:# or os.path.isfile (removed_np_arr_file) == False:

        print ("Remove outliers from the original array")
        dataset = np.concatenate ((total_data, total_label, total_car_ident_code, act_ad_date, total_rel_err), axis=1)

        # Get the dataset after removing removal_percent no. data points with the highest relative error from the total dataset
        len_dataset = total_data.shape[0]
        remain_len = int (len_dataset * (1 - removal_percent / 100.0) + 0.5)

        # Get the indexes of the dataset sorted by rel_err (the last column), (remain_len - 1) data points with rel_err < (remain_len)th data point's rel_err
        new_idx = np.argpartition (dataset[:,-1], remain_len)

        # The indexes of (remain_len) data points with the smallest rel_err.
        idx = new_idx [:remain_len]
        new_dataset = dataset [idx]

        # Sort by adv_date
        new_dataset = new_dataset [new_dataset[:, -2].argsort()]

        # The new dataset will store car_ident_code at the end of the matrix
        new_dataset = new_dataset[:, :-2] 

        # Save the dataset into a binary file
        #np.save (stored_np_arr_file, new_dataset)
        # Convert numpy array to dataframe format and save the dataset into a pytable file (compressed)
        #df_dataset = pd.DataFrame (new_dataset)
        #df_dataset.to_hdf (stored_np_arr_file, key)       

        return new_dataset 

        #else:
        #    print ("Remove ouliers by reloading the preprocessed data:", stored_np_arr_file)
        #    return np.load (stored_np_arr_file + ".npy")
        #    df_dataset = pd.read_hdf (stored_np_arr_file, key)

    def retrain_car2vect_after_remove_outliers (self, train_data, train_label, test_data, test_label, test_car_ident, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, removal_percent):
        train_data_remain = train_data [:, 0:d_remain]
        train_data_ident = train_data [:, d_remain:]
        test_data_remain = test_data [:, 0:d_remain]
        test_data_ident = test_data [:, d_remain:]

        # First train the model on the original train data (can remove a part of outliers previously)
        os.system ("mkdir -p ../checkpoint/rm_outliers_NN/car2vect/regressor1")
        model_path = self.model_dir + "/rm_outliers_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}_{5}".format (1, dataset_size, self.model_name, self.label, self.no_neuron_embed, self.no_neuron)
        y_predict_file_name_ = y_predict_file_name + "_{0}".format (1)
        mean_error_file_name_ = mean_error_file_name + "_{0}".format (1)
        x_ident_file_name_ = x_ident_file_name + "_{0}".format (1)
        x_embed_file_name_ = x_embed_file_name + "_{0}".format (1)
        print ("\n\n===========Predictor1")
        best_epoch = 33
        #best_epoch = self.car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, test_car_ident=test_car_ident, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func=self.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_, x_ident_file_name=x_ident_file_name_, x_embed_file_name=x_embed_file_name_, retrain=0)
        print ("Best epoch: ", best_epoch)
        
        # Restore the trained model
        # When restore model with the whole dataset, it can cause the error: Resource exhausted 
        # Devide the train set into smaller subsets (Eg. 5 subsets), push them to the model and concatenate the predictions later
        # TODO: change the "model_dir" arg to automatically set the directory
        meta_file = model_path + "_{0}.meta".format (best_epoch)
        ckpt_file = model_path + "_{0}.meta".format (best_epoch)

        (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val, total_arr_relative_err) = self.batch_computation_car2vect (5, train_data, train_label, d_ident, d_remain, meta_file, ckpt_file)
        line = np.zeros(len (train_label), dtype=[('truth', float), ('pred', float)])
        line['truth'] = train_label.reshape (train_label.shape[0])
        line['pred'] = predicted_train_label.reshape (predicted_train_label.shape[0])
        np.savetxt (y_predict_file_name + "_train_before_remove_outliers", line, fmt="%.2f\t%.2f")

        # Remove outliers from the train dataset baased on the train error
        new_train_set = self.remove_outliers (train_data, train_label, total_arr_relative_err, removal_percent)

        # Train the car2vect model based on the new dataset
        # Firstly, create the necessary data for car2vect model
        new_train_data = new_train_set [:, :train_data.shape[1]] 
        new_train_label = new_train_set [:, train_data.shape[1]:]

        tf.reset_default_graph ()
        os.system ("mkdir -p ../checkpoint/rm_outliers_NN/car2vect/regressor2")
        model_path = self.model_dir + "/rm_outliers_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}_{5}".format (2, dataset_size, self.model_name, self.label, self.no_neuron_embed, self.no_neuron)
        y_predict_file_name_ = y_predict_file_name + "_{0}".format (2)
        mean_error_file_name_ = mean_error_file_name + "_{0}".format (2)
        x_ident_file_name_ = x_ident_file_name + "_{0}".format (2)
        x_embed_file_name_ = x_embed_file_name + "_{0}".format (2)
        
        print ("\n\n===========Predictor2")
        best_epoch = self.car2vect (train_data=new_train_data, train_label=new_train_label, test_data=test_data, test_label=test_label, test_car_ident=test_car_ident, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func=self.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_, x_ident_file_name=x_ident_file_name_, x_embed_file_name=x_embed_file_name_, retrain=1)
        print ("Best epoch: ", best_epoch)

        # Apply the trained model to the test data
        # Restore the trained model
        # When restore model with the whole dataset, it can cause the error: Resource exhausted 
        # Devide the train set into smaller subsets (Eg. 5 subsets), push them to the model and concatenate the predictions later
        """meta_file = model_path + "_" + str (best_epoch) + ".meta"
        ckpt_file = model_path + "_" + str (best_epoch)
        (new_predicted_train_label, new_train_rmse_val, new_train_mae_val, new_train_relative_err_val, new_train_smape_val, new_total_arr_relative_err) = self.batch_computation_car2vect (5, new_train_data, new_train_label, d_ident, d_remain, meta_file, ckpt_file)
        sys.exit (-1)
        line = np.zeros(len (new_train_label), dtype=[('truth', float), ('pred', float)])
        line['truth'] = new_train_label.reshape (new_train_label.shape[0])
        line['pred'] = new_predicted_train_label.reshape (new_predicted_train_label.shape[0])
        np.savetxt (y_predict_file_name + "_train_after_remove_" + str (removal_percent), line, fmt="%.2f\t%.2f")"""

    def retrain_car2vect_from_total_set (self, total_data, total_label, total_car_ident_code, act_adv_date, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, removal_percent, ensemble_flag):
        """
            - Purpose: 
                + Train the model car2vect with the whole dataset, and then remove the data points with removal_percent highest relative error.
                + Sort the remaining dataset by act_adv_date, and divide it into train and test sets
                + Retrain the model car3vect with the new training data (if the "retrain" flag == 1 -> use the initial weights from the 1st train, 
                otherwise retrain from scratch.
        """
        # First train the model on the original train data (can remove a part of outliers previously)
        os.system ("mkdir -p ../checkpoint/rm_outliers_total_set_NN/car2vect/regressor1")
        model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}_{5}_total_set".format (1, dataset_size, self.model_name, self.label, self.no_neuron_embed, self.no_neuron)

        print ("\n\n===========Train total set")
        # If comment the below line, you need to check the checkpoint file in regressor1 (it should be compatible with the dataset) 
        #self.train_car2vect(train_data=total_data, train_label=total_label, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func="rel_err", model_path=model_path)
        
        # Restore the trained model
        # When restore model with the whole dataset, it can cause the error: Resource exhausted 
        # Devide the train set into smaller subsets (Eg. 5 subsets), push them to the model and concatenate the predictions later
        # TODO: change the "model_dir" arg to automatically set the directory
        meta_file = model_path + ".meta"
        ckpt_file = model_path

        (predicted_total_label, total_rmse_val, total_mae_val, total_relative_err_val, total_smape_val, total_arr_relative_err) = self.batch_computation_car2vect (5, total_data, total_label, d_ident, d_remain, meta_file, ckpt_file)
        total_np_arr = np.concatenate ((total_car_ident_code, act_adv_date, total_label, predicted_total_label), axis=1)
        total_df = pd.DataFrame (total_np_arr)
        np.savetxt (y_predict_file_name + "_total_before_remove_outliers", total_df, fmt="%d\t%d\t%d\t%d\t%s\t%.0f\t%d\t%.2f")

        # Remove outliers from the total dataset based on the relative error from the first train, on the other hand sort the dataset by act_adv_date
        # TODO: save this dataset
        stime = time.time()
        new_total_set = self.remove_outliers_total_set (total_data, total_label, total_car_ident_code, act_adv_date, total_arr_relative_err, dataset_size, removal_percent)
        print ("Time for remove outliers from dataset: %.3f" % (time.time() - stime))

        # Train the car2vect model based on the new dataset
        # Firstly, create the necessary data for car2vect model
        shape1_train = total_data.shape[1]
        new_total_data  = new_total_set [:, :shape1_train] 
        new_total_label = new_total_set [:, shape1_train:shape1_train+1]
        new_total_car_ident_code = new_total_set [:, shape1_train+1:]
        
        len_total_set   = new_total_data.shape[0]    
        data_training_percentage = 0.8
        len_train   = int (0.5 + len_total_set * data_training_percentage)
        new_train_data     = new_total_data[:len_train, :]
        new_train_label    = new_total_label[:len_train]
        new_test_data      = new_total_data[len_train:, :]
        new_test_label     = new_total_label[len_train:]
        new_test_car_ident = new_total_car_ident_code[len_train:, :]

        print ("Old dataset:", total_data.shape, total_label.shape)
        print ("New dataset:", new_total_data.shape, new_total_label.shape)
        print ("New train dataset:", new_train_data.shape, new_train_label.shape)
        print ("New test dataset:", new_test_data.shape, new_test_label.shape)
        print ("New test_car_ident:", new_test_car_ident.shape)

        if ensemble_flag == 0:
            tf.reset_default_graph ()
            os.system ("mkdir -p ../checkpoint/rm_outliers_total_set_NN/car2vect/regressor2")
            model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}_{5}_total_set".format (2, dataset_size, self.model_name, self.label, self.no_neuron_embed, self.no_neuron)
            y_predict_file_name_ = y_predict_file_name + "_2"
            mean_error_file_name_ = mean_error_file_name + "_2"
            x_ident_file_name_ = x_ident_file_name + "_2"
            x_embed_file_name_ = x_embed_file_name + "_2"
            
            print ("\n\n===========Predictor2")
            best_epoch = self.car2vect (train_data=new_train_data, train_label=new_train_label, test_data=new_test_data, test_label=new_test_label, test_car_ident=new_test_car_ident, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func=self.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_, x_ident_file_name=x_ident_file_name_, x_embed_file_name=x_embed_file_name_, retrain=1) # if use retrain=1 -> initialize weights from the previous model
            print ("Best epoch: ", best_epoch)

            """#####################
            meta_file = model_path + "_" + str (best_epoch) + ".meta"
            ckpt_file = model_path + "_" + str (best_epoch) 
            test_data_remain = new_test_data [:, 0:d_remain]
            test_data_ident = new_test_data [:, d_remain:]
            (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_car2vect (test_data_ident, test_data_remain, new_test_label, meta_file, ckpt_file, train_flag=0)
            #####################"""

        elif ensemble_flag == 2:
           self.gradient_boosting_NN_car2vect (new_train_data, new_train_label, new_test_data, new_test_label, new_test_car_ident, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, retrain=2) 

        elif ensemble_flag == 5:
            self.bagging_NN_car2vect (new_train_data, new_train_label, new_test_data, new_test_label, new_test_car_ident, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, self.sample_ratio, retrain=1)

        else:
            raise ValueError ("[retrain_car2vect_from_total_set] This model is not supported!")

    def batch_computation_car2vect  (self, no_batch, train_data, train_label, d_ident, d_remain, meta_file, ckpt_file):
        train_data_remain = train_data [:, 0:d_remain]
        train_data_ident = train_data [:, d_remain:]
        # devide the train set into smaller subsets, push them to the model and concatenate them later
        total_se = 0
        total_ae = 0
        total_relative_err = 0
        total_smape = 0
        start_index = 0
        end_index = 0
        total_predicted_y = np.zeros ((0, 1))
        total_arr_relative_err = np.zeros ((0, 1))
        total_x_embed_val = np.zeros ((0, self.d_embed))
        len_train = len(train_data)
        batch_size = int (len_train / no_batch)
        train_total_batch = int (np.ceil (float (len_train)/batch_size))
        for i in range (train_total_batch):
            if len_train - end_index < batch_size:
                end_index = len_train
            else:
                end_index = (i+1) * batch_size

            batch_x_ident = train_data_ident [start_index : end_index]
            batch_x_remain = train_data_remain [start_index : end_index]
            batch_y = train_label [start_index : end_index]

            start_index = end_index

            (predicted_y, train_sum_se_val, train_sum_ae_val, train_sum_relative_err_val, train_sum_smape_val, arr_relative_err_val) = self.restore_model_car2vect (batch_x_ident, batch_x_remain, batch_y, meta_file, ckpt_file, train_flag=1)
            total_predicted_y = np.concatenate ((total_predicted_y, predicted_y))
            total_arr_relative_err = np.concatenate ((total_arr_relative_err, arr_relative_err_val))
            total_se += train_sum_se_val
            total_ae += train_sum_ae_val
            total_relative_err += train_sum_relative_err_val
            total_smape += train_sum_smape_val

        assert end_index == len_train   

        predicted_train_label = total_predicted_y
        train_rmse_val = np.sqrt (total_se/len_train)
        train_mae_val = total_ae/len_train
        train_relative_err_val = total_relative_err/len_train
        train_smape_val = total_smape/len_train
        print (total_arr_relative_err.shape, np.max (total_arr_relative_err), np.min (total_arr_relative_err), np.average (total_arr_relative_err), train_relative_err_val)
        return (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val, total_arr_relative_err)
    
    def gradient_boosting_NN_baseline_Tree (self, train_data, train_label, test_data, test_label, y_predict_file_name, mean_error_file_name, dataset_size):
        """
            - Purpose: Apply Gradient Boosting (a kind of ensemble method) using baseline NN, after that using Decision Tree to estimate 
        """
        train_label_copy = train_label.astype (np.float64)
        test_label_copy = test_label.astype (np.float64)
        list_predicted_test_label = []

        print ("\n\n==============Baseline NN:")
        loss_func = "rel_err"
        """model_path = self.model_dir + "/gb_NN/baseline/regressor" + str (i+1) + "/" + dataset_size + "_" + self.model_name  + "_" + self.label  + "_baseline_" + str (self.no_neuron) + "_" + str (self.no_hidden_layer) + "_" + loss_func
        y_predict_file_name_ = y_predict_file_name + "_" + str (i+1)
        mean_error_file_name_ = mean_error_file_name + "_" + str (i+1)

        best_epoch = self.baseline (train_data=train_data, train_label=train_label_copy, test_data=test_data, test_label=test_label_copy, no_neuron=no_neuron, no_hidden_layer = no_hidden_layer, loss_func=loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_)"""
        best_epoch = 9
        model_path = self.model_dir + "/gb_NN/baseline/regressor" + str (1) + "/" + dataset_size + "_" + self.model_name  + "_" + self.label  + "_baseline_2000_2_" + loss_func
        print ("Best epoch: ", best_epoch)

        meta_file = model_path + "_" + str (best_epoch) + ".meta"
        ckpt_file = model_path + "_" + str (best_epoch) 
        (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val) = self.restore_model_NN_baseline (train_data, train_label_copy, meta_file, ckpt_file)
        (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_NN_baseline (test_data, test_label_copy, meta_file, ckpt_file)
        print (predicted_test_label[:10])
        print (test_label_copy[:10])
        list_predicted_test_label.append (predicted_test_label)
        print ("=================================")

        #reg_tree = tree.DecisionTreeRegressor(criterion="mse", min_samples_split=5, max_depth=20) 
        print ("\n\n=====Gradient boosting tree:")
        reg_tree = ensemble.GradientBoostingRegressor(loss="ls", learning_rate=0.01, n_estimators=10, max_depth=10)
        train_label_copy -= predicted_train_label  
        test_label_copy -= predicted_test_label 
        reg_tree.fit (train_data, train_label_copy)

        predicted_train_label = self.get_predicted_label (reg_tree, train_data) 
        predicted_test_label = self.get_predicted_label (reg_tree, test_data) 
        print (predicted_test_label[:10])
        print (test_label_copy[:10])

        list_predicted_test_label.append (predicted_test_label)

        predicted_test_label = sum (pred_y for pred_y in list_predicted_test_label)
        (test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = get_err (predicted_test_label, test_label)
        print ("Final err:", test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val)
        
    # Create a random subsample from the dataset with replacement
    def subsample (self, data, label, ratio):
        
        #TODO: training data permutation
        dataset = np.concatenate ((data, label), axis = 1)
        n_sample = int (round(len(dataset) * ratio))

        dataset_shuffled = np.random.permutation(dataset)
        data_shuffled  = dataset_shuffled [:, 0:data.shape[1]]
        label_shuffled = dataset_shuffled [:, data.shape[1]:]
        return (data_shuffled[:n_sample], label_shuffled[:n_sample])

    def bagging_NN_baseline (self, train_data, train_label, test_data, test_label, y_predict_file_name, mean_error_file_name, dataset_size, ratio):
        """
            - Purpose: Apply bagging (a kind of ensemble method) using baseline NN.
        """
        list_predicted_test_label = []
        np.random.seed (1)

        for i in range (self.num_regressor):
            print ("\n\n==============regressor%d" %(i+1))
            (X_train, y_train) = self.subsample (train_data, train_label, ratio)
            model_path = self.model_dir + "/bagging_NN/baseline/regressor" + str (i+1) + "/" + dataset_size + "_" + self.model_name  + "_" + self.label  + "_baseline_" + str (self.no_neuron) + "_" + str (self.no_hidden_layer)
            y_predict_file_name_ = y_predict_file_name + "_" + str (i+1)
            mean_error_file_name_ = mean_error_file_name + "_" + str (i+1)

            # Reset to the default graph, avoid to the error of redefining variables
            tf.reset_default_graph ()

            # Training
            best_epoch = self.baseline (train_data=X_train, train_label=y_train, test_data=test_data, test_label=test_label, no_neuron=self.no_neuron, no_hidden_layer=self.no_hidden_layer, loss_func="rel_err", model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_)
            bash_cmd = "cd ../checkpoint/bagging_NN/baseline/regressor" + str (i+1) + "; mkdir temp_save; rm temp_save/*; cp checkpoint *_" + str (best_epoch) + ".*" + " temp_save; cd ../../../../Code"
            print ("bash_cmd:", bash_cmd)
            os.system (bash_cmd)
            print ("Best epoch: ", best_epoch)
            meta_file = model_path + "_" + str (best_epoch) + ".meta"
            ckpt_file = model_path + "_" + str (best_epoch) 
            (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_NN_baseline (test_data, test_label, meta_file, ckpt_file)
            list_predicted_test_label.append (predicted_test_label)
            print ("predicted_test_label (" + str(i) + ")", predicted_test_label[:10])
            print ("=================================")

        print ("label", test_label[:10])
        for i in range (self.num_regressor):
            predicted_test_label = sum (pred_y for pred_y in list_predicted_test_label[:i+1]) / (i+1)
            print ("predicted_test_label (" + str(i) + ")", predicted_test_label[:10])
            (test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = get_err (predicted_test_label, test_label)
            print ("Err after " + str (i+1)  + " regressors:", test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val)
            line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
            line['truth'] = test_label.reshape (test_label.shape[0])
            line['pred'] = predicted_test_label.reshape (predicted_test_label.shape[0])
            np.savetxt (y_predict_file_name_ + "_final" + str(i), line, fmt="%.2f\t%.2f")

    def bagging_NN_car2vect (self, train_data, train_label, test_data, test_label, test_car_ident, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, ratio, retrain):
        """
            - Purpose: Apply bagging (a kind of ensemble method) using car2vect.
        """
        train_data_remain = train_data [:, 0:d_remain]
        train_data_ident = train_data [:, d_remain:]
        test_data_remain = test_data [:, 0:d_remain]
        test_data_ident = test_data [:, d_remain:]
        list_predicted_test_label = []
        np.random.seed (1)

        for i in range (self.num_regressor):
            print ("\n\n==============regressor%d" %(i+1))
            (X_train, y_train) = self.subsample (train_data, train_label, ratio)
            model_path = self.model_dir + "/bagging_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}_{5}".format (i+1, dataset_size, self.model_name, self.label, self.no_neuron_embed, self.no_neuron)
            y_predict_file_name_ = y_predict_file_name + "_" + str (i+1)
            mean_error_file_name_ = mean_error_file_name + "_" + str (i+1)
            x_ident_file_name_ = x_ident_file_name + "_" + str (i+1)
            x_embed_file_name_ = x_embed_file_name + "_" + str (i+1)

            # Reset to the default graph, avoid to the error of redefining variables
            tf.reset_default_graph ()

            # Training
            os.system ("mkdir -p ../checkpoint/bagging_NN/car2vect/regressor" + str (i+1)) # TODO: move this line to Main.py
            
            best_epoch = self.car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, test_car_ident=test_car_ident, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func="rel_err", model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_, x_ident_file_name=x_ident_file_name_, x_embed_file_name=x_embed_file_name_, retrain=retrain)
            bash_cmd = "cd ../checkpoint/bagging_NN/car2vect/regressor" + str (i+1) + "; mkdir temp_save; rm temp_save/*; cp checkpoint *_" + str (best_epoch) + ".*" + " temp_save; cd ../../../../Code"
            print ("bash_cmd:", bash_cmd)
            os.system (bash_cmd)
            print ("Best epoch: ", best_epoch)
            meta_file = model_path + "_" + str (best_epoch) + ".meta"
            ckpt_file = model_path + "_" + str (best_epoch) 
            (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_car2vect (test_data_ident, test_data_remain, test_label, meta_file, ckpt_file, train_flag=0)
            list_predicted_test_label.append (predicted_test_label)
            print ("predicted_test_label (" + str(i) + ")", predicted_test_label[:10])
            print ("=================================")

        print ("label", test_label[:10])
        for i in range (self.num_regressor):
            predicted_test_label = sum (pred_y for pred_y in list_predicted_test_label[:i+1]) / (i+1)
            print ("predicted_test_label (" + str(i) + ")", predicted_test_label[:10])
            (test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = get_err (predicted_test_label, test_label)
            print ("Err after " + str (i+1)  + " regressors:", test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val)
            line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
            line['truth'] = test_label.reshape (test_label.shape[0])
            line['pred'] = predicted_test_label.reshape (predicted_test_label.shape[0])
            np.savetxt (y_predict_file_name_ + "_final" + str(i), line, fmt="%.2f\t%.2f")



        
