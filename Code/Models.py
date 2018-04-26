
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

    """def kmean_cluster (self, train_data, test_data)
        kmeans = KMeans(n_clusters=1000, random_state=0).fit(train_data)
        train_label = kmeans.labels_
        test_label = kmeans.predict (test_data)"""

    def knn (self, train_data, train_label, test_data, test_label):
        stime = time.time()
        print ("Start KNN...")
        for n in [10]:#2, 5, 10, 20, 30, 40, 50, 100]:
            print ("====n_neighbors:", n)
            nn = KNeighborsRegressor(n_neighbors=n)
            nn.fit (train_data, train_label)
            predicted_test_label = nn.predict (test_data)
            rmse, mae, rel_err, smape = get_err (predicted_test_label, test_label)
            print ("[knn] (rmse,mae,rel_err,smape):", rmse, mae, rel_err, smape)
            print ("Time for K-Nearest neighbor: %.3f" % (time.time() - stime))        


    def sklearn_regression (self, reg_type, train_data, train_label, test_data, test_label):
        """
            Use GridSearch to find the best hyper-parameters for each model
        """
        ###########
        ### test
        """regr = linear_model.LinearRegression ()
        regr.fit (train_data, train_label)
        pred_test_label = regr.predict (test_data)
        pred_test_label = [0 if pred_test_label[i] < 0 else min (pred_test_label[i], 3) for i in range (len (pred_test_label))] 
        print ("rmse:", np.sqrt (mean_squared_error (test_label, pred_test_label)))
        sys.exit (-1)
        ###########"""
        stime = time.time()
        print ("\n\n#########################")
        print ("Model:", reg_type)
        print ("#########################")
        grid_file = self.grid_file
        grid_file += reg_type
        cv_num = 5

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
            param_grid = [ {"criterion":["mse"], "min_samples_split":[10], "max_depth":[20]}]

        elif reg_type == "GradientBoostingRegressor":
            regr = ensemble.GradientBoostingRegressor()
            #param_grid = [ {"n_estimators": [10, 100, 500, 1000],"learning_rate": [0.001, 0.01, 0.1, 0.2], "max_depth":[10,20,30]}]
            param_grid = [ {"n_estimators": [10],"learning_rate": [0.001, 0.01], "max_depth":[10, 20]}]

        elif reg_type == "AdaBoostRegressor":
            regr = ensemble.AdaBoostRegressor()
            #param_grid = [ {"n_estimators": [10, 100, 500, 1000], "min_samples_split": [5, 10, 20], "max_depth": [10, 20, 30]}]
            param_grid = [ {"n_estimators": [10]}]

        elif reg_type == "RandomForestRegressor":
            regr = ensemble.RandomForestRegressor()
            #param_grid = [ {"n_estimators": [10, 100, 500, 1000], "min_samples_split": [5, 10, 20], "max_depth": [10, 20, 30]},
            #               {"bootstrap": [False], "n_estimators": [10, 100, 500, 1000], "min_samples_split": [5, 10, 20], "max_depth": [10, 20, 30]} 
            #]
            param_grid = [ {"n_estimators": [10], "min_samples_split": [10], "max_depth": [10, 20]},
                           {"bootstrap": [False], "n_estimators": [10], "min_samples_split": [10], "max_depth": [10, 20]} 
            ]
        else:
            raise ValueError ("This model is not supported!")

        err = make_scorer (get_rmse, greater_is_better=False)

        print ("========= Training...")
        if os.path.isfile (grid_file) == False:
            grid_search = GridSearchCV (regr, param_grid, cv = cv_num, scoring = err) #err, "neg_mean_squared_error"
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
        #predicted_train_label = [0 if predicted_train_label[i] < 0 else min (predicted_train_label[i], 400) for i in range (len (predicted_train_label))] 

        (train_rmse, train_mae, train_rel_err, train_smape) = get_err (predicted_train_label, train_label)
        
        predicted_test_label = self.get_predicted_label (grid_search, test_data) 
        #predicted_test_label = [0 if predicted_test_label[i] < 0 else min (predicted_test_label[i], 400) for i in range (len (predicted_test_label))] 
        predicted_test_label = np.clip (a=predicted_test_label, a_min=0, a_max=np.max (predicted_train_label))

        (test_rmse, test_mae, test_rel_err, test_smape) = get_err (predicted_test_label, test_label) 

        print (np.c_ [test_label, predicted_test_label])
        print (np.c_ [np.min (predicted_train_label), np.max (predicted_train_label)])
        print (np.c_ [np.min (test_label), np.max (test_label)])
        print (np.c_ [np.min (predicted_test_label), np.max (predicted_test_label)])

        print (reg_type + ": (rmse, mae, rel_err, smape)")
        print (" Train err: (%.3f, %.3f, %.3f %%, %.3f %%)" % (train_rmse, train_mae, train_rel_err, train_smape))
        print (" Test err : (%.3f, %.3f, %.3f %%, %.3f %%)" % (test_rmse, test_mae, test_rel_err, test_smape) )

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

        self.test_uncertainty = args.test_uncertainty
        self.test_pred_time = args.test_pred_time

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

        # Regularization hyper-parameters
        self.gama = 1e-4
        self.alpha = 1e-3
        self.beta = 1e-3

    
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
            net = slim.fully_connected (net, no_unit, scope='hidden_layer'+str(i+1), activation_fn=tf.nn.relu)  
            net = slim.dropout (net, self.dropout, scope='dropout'+str(i+1))

        prediction = slim.fully_connected (net, 1, scope='output_layer', activation_fn=tf.nn.relu) #None) #, reuse=tf.AUTO_REUSE)
        tf.identity (prediction, name="prediction")

        return X, Y, prediction

    def restore_model_NN_baseline (self, data, label, meta_file, ckpt_file, train_flag):
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
                arr_se = graph.get_tensor_by_name ("arr_se:0")
                arr_ae = graph.get_tensor_by_name ("arr_ae:0")
                arr_rel_err = graph.get_tensor_by_name ("arr_rel_err:0")
                sum_smape = graph.get_tensor_by_name ("sum_smape:0")

                predicted_y, sum_se_val, sum_ae_val, sum_rel_err_val, sum_smape_val, arr_se_val, arr_ae_val, arr_rel_err_val = sess.run([prediction, sum_se, sum_ae, sum_rel_err, sum_smape, arr_se, arr_ae, arr_rel_err], feed_dict)
                return (predicted_y, sum_se_val, sum_ae_val, sum_rel_err_val, sum_smape_val, arr_se_val, arr_ae_val, arr_rel_err_val)


    
    def train_baseline (self, train_data, train_label, no_neuron, no_hidden_layer, loss_func, model_path): 
       
        #building car embedding model
        X, Y, prediction = self.build_model(train_data.shape[1], no_neuron, no_hidden_layer) 

        # Try to use weights decay
        num_batches_per_epoch = int(len(train_data) / self.batch_size)
        decay_steps = int(num_batches_per_epoch * self.decay_step)
        global_step = tf.Variable(0, trainable = False)
        learning_rate = self.learning_rate #tf.train.exponential_decay(self.learning_rate, global_step, decay_steps, self.decay_rate, staircase = True)

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
        sum_se = tf.reduce_sum (tf.squared_difference(prediction, Y_), name = "sum_se")
        sum_ae = tf.reduce_sum (tf.abs (prediction - Y_), name = "sum_ae")
        sum_relative_err = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y_), Y_)), 100, name = "sum_rel_err")

        arr_se =tf.squared_difference(prediction, Y_, name = "arr_se")
        arr_ae = tf.abs (prediction - Y_, name = "arr_ae")
        arr_rel_err = tf.multiply (tf.divide (tf.abs (prediction - Y_), Y_), 100, name = "arr_rel_err")

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
            train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
            train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

            scaler = MinMaxScaler(feature_range=(self.min_price, self.max_price))
            scaler.fit (train_label_shuffled)
            if self.scale_label == 1:
                train_label_shuffled_scaled = scaler.transform (train_label_shuffled)
            else:
                train_label_shuffled_scaled = train_label_shuffled

            len_train = len(train_data)
            train_total_batch = int (np.ceil (float (len_train)/self.batch_size))

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
                
                if epoch == self.epoch-1 or (epoch + 1)%10 == 0: 
                    ckpt_file = model_path
                    save_path = saver.save (sess, ckpt_file) #"../checkpoint/baseline/test4")#model_path + "_" + str (epoch)) 
                    print('Model saved in file: %s' % save_path)

                # Training data permutation
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
            

    def baseline (self, train_data, train_label, test_data, test_label, total_car_ident, total_act_adv_date, total_sale_date, no_neuron, no_hidden_layer, loss_func, model_path, y_predict_file_name, x_ident_file_name, mean_error_file_name): # Used for 1train-1test
    #def baseline (self, train_data, train_label, test_data, test_label, model_path, X, Y, prediction, weights, fold): # used for Cross-validation 
       
        len_train = len(train_data)
        len_test  = len(test_data)
        test_car_ident = total_car_ident [len_train:, :] 
        test_act_adv_date = total_act_adv_date [len_train:, :] 
        test_sale_date = total_sale_date [len_train:, :] 

        if using_CV_flag == 0:
            X, Y, prediction = self.build_model(train_data.shape[1], no_neuron, no_hidden_layer) 
            mean_error_file_name_ = mean_error_file_name
            y_predict_file_name_ = y_predict_file_name
        else:
            mean_error_file_name_ = mean_error_file_name + "_" + "fold" + str (fold+1)
            y_predict_file_name_ = y_predict_file_name + "_" + "fold" + str (fold+1)

        identification = np.concatenate ((test_car_ident, test_act_adv_date, test_sale_date),axis=1)
        np.savetxt (x_ident_file_name, identification, fmt="%d\t%d\t%d\t%d\t%s\t%s\t%d")  

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
        sum_se = tf.reduce_sum (tf.squared_difference(prediction, Y_), name = "sum_se")
        sum_ae = tf.reduce_sum (tf.abs (prediction - Y_), name = "sum_ae")
        sum_relative_err = tf.multiply (tf.reduce_sum (tf.divide (tf.abs (prediction - Y_), Y_)), 100, name = "sum_rel_err")

        arr_se =tf.squared_difference(prediction, Y_, name = "arr_se")
        arr_ae = tf.abs (prediction - Y_, name = "arr_ae")
        arr_rel_err = tf.multiply (tf.divide (tf.abs (prediction - Y_), Y_), 100, name = "arr_rel_err")
        arr_smape = tf.multiply (tf.divide (tf.abs (prediction - Y_), tf.abs (Y_) + tf.abs (prediction)), 100, name = "arr_smape")

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
            train_data_shuffled = train_set_shuffled [:, 0:train_data.shape[1]]
            train_label_shuffled = train_set_shuffled [:, train_data.shape[1]:]

            scaler = MinMaxScaler(feature_range=(self.min_price, self.max_price))
            scaler.fit (train_label_shuffled)
            if self.scale_label == 1:
                train_label_shuffled_scaled = scaler.transform (train_label_shuffled)
            else:
                train_label_shuffled_scaled = train_label_shuffled

            train_total_batch = int (np.ceil (float (len_train)/self.batch_size))
            test_total_batch = int (np.ceil (float (len_test)/self.batch_size))
            smallest_epoch_test_err_val = 10e6
            best_epoch = 0

            epoch_list = [] 
            train_rmse_list = [] 
            train_mae_list = [] 
            train_rel_err_list = [] 
            train_smape_list = [] 
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
                train_rmse_list.append (epoch_train_rmse_val)
                train_mae_list.append (epoch_train_mae_val)
                train_rel_err_list.append (epoch_train_relative_err_val)
                train_smape_list.append (epoch_train_smape_val)
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
                    threshold_err = 7 #9.5# 
                    epoch_test_err_val = epoch_test_relative_err_val
                elif loss_func == "mae":
                    threshold_err = 2
                    epoch_test_err_val = epoch_test_mae_val
                elif loss_func == "mse":
                    threshold_err = 13.8
                    epoch_test_err_val = epoch_test_rmse_val

                #if (epoch + 1) % 1 == 0:
                if (epoch_test_err_val < threshold_err) or (epoch == self.epoch - 1):
                    np.savetxt (y_predict_file_name_ + "_" + str (epoch), line, fmt="%.2f\t%.2f")
                    print ("Pred_label saved in file:", y_predict_file_name_ + "_" + str (epoch))
                    save_path = saver.save (sess, model_path + "_" + str (epoch)) #, global_step=global_step)
                    print("Model saved in file:", save_path)

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

    def build_car2vect_model (self, no_neuron, no_neuron_embed, d_ident, d_embed, d_remain):
        """
            - Args:
                + no_neuron: the number of neuron in 'main' hiddenlayers.
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
        output1_ = slim.dropout (output1, self.dropout, scope='dropout1')
        #output2 = slim.fully_connected (output1_, no_neuron_embed, scope='hidden_embed2', activation_fn=tf.nn.relu)
        #output2_ = slim.dropout (output2, self.dropout, scope='dropout2')
        x_embed = slim.fully_connected (output1_, d_embed, scope='output_embed', activation_fn=None, weights_initializer=he_init) #, activation_fn=tf.nn.relu)#, activation_fn=None) # 3-dimension of embeding NN
        #x_embed = slim.fully_connected (output1, d_embed, scope='output_embed', activation_fn=tf.nn.relu) # 3-dimension of embeding NN
        #x_embed = slim.fully_connected (output1, d_embed, scope='output_embed') # seperate the activation function to another step to use batch normalization.
        #x_embed = self.batch_norm (x_embed, phase_train) # batch normalization
        #x_embed = tf.nn.elu (x_embed, name="elu_output_embed")

        #mean, var = tf.nn.moments (x_embed, [0], keep_dims=True)
        #x_embed = tf.div(tf.subtract(x_embed, mean), tf.sqrt(var))
        #x_embed = tf.div (x_embed - tf.reduce_min (x_embed), tf.reduce_max (x_embed) - tf.reduce_min (x_embed))

        input3 = tf.concat ([x_remain, x_embed], 1)

        output3 = slim.fully_connected(input3, no_neuron, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=he_init)
        output3_ = slim.dropout (output3, self.dropout, scope='dropout3')
        #output4 = slim.fully_connected(output3_, no_neuron, scope='hidden_main_2', activation_fn=tf.nn.relu)
        #output4_ = slim.dropout (output4, self.dropout, scope='dropout3')
        prediction = slim.fully_connected(output3_, 1, scope='output_main', activation_fn=tf.nn.relu, weights_initializer=he_init) #tf.nn.relu) #None) # 1-dimension of output NOTE: only remove relu activation function in the last layer if using Gradient Boosting, because the differece can be negative (default activation function of fully_connected is relu))
        tf.identity (prediction, name="prediction")

        return x_ident, x_remain, Y, x_embed, prediction, phase_train

    def build_car2vect_model_retrained (self, no_neuron, no_neuron_embed, d_ident, d_embed, d_remain, train_data, train_label, meta_file, ckpt_file):
        """
            Purpose: Use weights and biases from pre-trained model as initializations
        """
        train_data_remain = train_data [:, 0:d_remain]
        train_data_ident = train_data [:, d_remain:]
        (init_w_hid_embed_1_val, init_bias_hid_embed_1_val, init_w_out_embed_val, init_bias_out_embed_val, init_w_hid_main_1_val, init_bias_hid_main_1_val, init_w_out_main_val, init_bias_out_main_val) = self.restore_weights_car2vect (train_data_ident, train_data_remain, train_label, meta_file, ckpt_file) # TODO: ???why if I leave this line after placeholder initialization, it causes the error: some nodes: x_ident_1, x_remain_1, Y_1, phase_train_1 will be created in the graph?

        tf.reset_default_graph () # This line is really important, if I restore weights from the pre-trained model, and don't reset the graph to default, the results are weird!
        x_ident = tf.placeholder(tf.float32, [None, d_ident], name="x_ident")
        x_remain = tf.placeholder(tf.float32, [None, d_remain], name="x_remain")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        print ("build_car2vect_model: d_ident:", d_ident, "d_remain:", d_remain, "d_embed:", d_embed, "no_neuron_embed:", no_neuron_embed, "no_neuron_main:", no_neuron)
        
        output1 = slim.fully_connected (x_ident, no_neuron_embed, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=tf.constant_initializer (init_w_hid_embed_1_val), biases_initializer=tf.constant_initializer (init_bias_hid_embed_1_val)) #None) #
        output1_ = slim.dropout (output1, self.dropout, scope='dropout1')
        output2 = slim.fully_connected (output1_, no_neuron_embed, scope='hidden_embed2', activation_fn=tf.nn.relu)
        output2_ = slim.dropout (output2, self.dropout, scope='dropout2')
        x_embed = slim.fully_connected (output2_, d_embed, scope='output_embed', activation_fn=None, weights_initializer=tf.constant_initializer (init_w_out_embed_val), biases_initializer=tf.constant_initializer (init_bias_out_embed_val)) 

        input3 = tf.concat ([x_remain, x_embed], 1)

        output3 = slim.fully_connected(input3, no_neuron, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=tf.constant_initializer (init_w_hid_main_1_val), biases_initializer=tf.constant_initializer (init_bias_hid_main_1_val))
        output3 = slim.dropout (output3, self.dropout, scope='dropout3')
        #prediction = slim.fully_connected(output3, 1, scope='output_main', activation_fn=tf.nn.relu, weights_initializer=tf.constant_initializer (init_w_out_main_val), biases_initializer=tf.constant_initializer (init_bias_out_main_val)) #None) # 1-dimension of output
        prediction = slim.fully_connected(output3, 1, scope='output_main', activation_fn=None, weights_initializer=tf.constant_initializer (init_w_out_main_val), biases_initializer=tf.constant_initializer (init_bias_out_main_val)) #None) # 1-dimension of output
        tf.identity (prediction, name="prediction")

        return x_ident, x_remain, Y, x_embed, prediction, phase_train
    
    def build_car2vect_regul_model (self, no_neuron, no_neuron_embed, d_ident, d_embed, d_remain):
        """
            - Args:
                + no_neuron: the number of neuron n 'main' hiddenlayers.
                + no_neuron_embed: the number of neuron in 'embedding' hiddenlayers.
                + d_ident: the dimension of the one-hot vector of a car identification (manufacture_code, ..., rating_code)
                + d_remain: the dimension of the remaining features (after one-hot encoding)
            - Purpose: Create the model of NN with using car2vect: from the one-hot encode of a car identification, use word2vect to embed it into a vector with small dimension.
        """
       
        def cosine_sim (a1, a2):
            normalized_a1 = tf.nn.l2_normalize (a1, 1)        
            normalized_a2 = tf.nn.l2_normalize (a2, 1)
            d1 = tf.losses.cosine_distance (normalized_a1, normalized_a2, 0)
            #d2 = tf.reduce_sum (tf.multiply (normalized_a1, normalized_a2))
            return 1-d1 #1-d1 # d2

        def cosine_dist_1tensor (a):
            normalized_a = tf.nn.l2_normalize (a, 1)        
            prod = tf.matmul (normalized_a, normalized_a, adjoint_b=True)
            dist = 1 - prod
            return tf.reduce_sum (dist, [0, 1]) / 2.0

        x_ident = tf.placeholder(tf.float32, [None, d_ident], name="x_ident")
        car_ident = tf.placeholder(tf.float32, [None, 5], name="car_ident")
        x_remain = tf.placeholder(tf.float32, [None, d_remain], name="x_remain")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        print ("build_car2vect_model: d_ident:", d_ident, "d_remain:", d_remain, "d_embed:", d_embed, "no_neuron_embed:", no_neuron_embed, "no_neuron_main:", no_neuron)
        rep_model_code = tf.reshape (car_ident[:, 1], [-1, 1])
        len_data = tf.shape (rep_model_code)[0]

        # He initializer
        # Embeding NN
        he_init = tf.contrib.layers.variance_scaling_initializer ()
        output1 = slim.fully_connected (x_ident, no_neuron_embed, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=he_init, weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta))
        output1_ = slim.dropout (output1, self.dropout, scope='dropout1')

        output2 = slim.fully_connected (output1_, no_neuron_embed, scope='hidden_embed2', activation_fn=tf.nn.relu, weights_initializer=he_init, weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta)) #None) #
        output2_ = slim.dropout (output2, self.dropout, scope='dropout2')

        x_embed = slim.fully_connected (output2_, d_embed, scope='output_embed', activation_fn=None, weights_initializer=he_init, weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta)) # 3-dimension of embeding NN

        input3 = tf.concat ([x_remain, x_embed], 1)

        # Main NN
        output3 = slim.fully_connected(input3, no_neuron, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=he_init, weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta))
        output3_ = slim.dropout (output3, self.dropout, scope='dropout3')

        output4 = slim.fully_connected(output3_, no_neuron, scope='hidden_main2', activation_fn=tf.nn.relu, weights_initializer=he_init, weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta))
        output4_ = slim.dropout (output4, self.dropout, scope='dropout4')

        prediction = slim.fully_connected(output4_, 1, scope='output_main', activation_fn=None, weights_initializer=he_init, weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta)) #tf.nn.relu) #None) # 1-dimension of output NOTE: only remove relu activation function in the last layer if using Gradient Boosting, because the differece can be negative (default activation function of fully_connected is relu))
        tf.identity (prediction, name="prediction")
        
        ####################################
        # Regularization
        ###################################
        regul1 = tf.reduce_sum (tf.get_collection (tf.GraphKeys.REGULARIZATION_LOSSES))

        # Cluster
        x_embed_add_rep = tf.concat ([rep_model_code, x_embed], 1)
        # Sort by rep_model_code
        x_embed_add_rep_sorted = tf.gather (x_embed_add_rep, tf.nn.top_k (x_embed_add_rep [:, 0], k=len_data).indices) 
        rep_model_unique, idx, count = tf.unique_with_counts (x_embed_add_rep_sorted[:, 0])

        # Compute the centroid vectors of every vector in each cluster (rep_model)
        x_embed_mean = tf.segment_mean (x_embed_add_rep_sorted, idx) # the 1st column is rep_model_code, the 3 other columns are x_embed_mean
        # Remove 1st column
        x_embed_mean = x_embed_mean [:, 1:]

        # Normalize all of the centroid vector
        norm = tf.norm (x_embed_mean, 1)
        x_embed_mean_normed = x_embed_mean #/ norm

        # Expand to a vector, compatible with the index of x_embed_sorted
        centroid_normed = tf.gather (x_embed_mean_normed, idx)

        # Remove 1st column
        x_embed_sorted = x_embed_add_rep_sorted [:, 1:]

        # The similiraty of every point with its centroid is added cummulatively 
        #regul_gather = tf.reduce_sum (cosine_sim (x_embed_sorted, centroid)) # want to keep this value large
        # The distance of every point with its centroid is added cummulatively
        regul_gather = tf.sqrt (tf.reduce_sum ((x_embed_sorted - centroid_normed) ** 2)) # want to keep this value small 

        # The distance between every centroid is added cummulatively 
        regul_spread = cosine_dist_1tensor (x_embed_mean_normed) # want to keep this value large

        """# Apply a NN to get the regularization item for the loss function
        regul_in = tf.stack ([[regul_gather], [regul_spread]])
        regul_out1 = slim.fully_connected (regul_in, 1000, scope='hidden_regul', activation_fn=tf.nn.relu, weights_initializer=he_init)
        regul_out = slim.fully_connected (regul_out1, 1, scope='out_regul', activation_fn=None, weights_initializer=he_init)"""

        regul_out = self.gama * regul1 + self.alpha * regul_gather - self.beta * regul_spread #+ self.beta * 1 / regul_spread
        #regul_out = regul1 - self.alpha * regul_gather * regul_spread
        ####################################

        return x_ident, x_remain, Y, x_embed, prediction, phase_train, car_ident, regul1, regul_gather, regul_spread, regul_out #, rep_model_code, centroid, x_embed_add_rep_sorted, regul_gather, regul_spread, regul_out

    def build_car2vect_regul_model_retrained (self, no_neuron, no_neuron_embed, d_ident, d_embed, d_remain, train_data, train_label, train_car_ident, meta_file, ckpt_file):
        """
            Purpose: Use weights and biases from pre-trained model as initializations
        """
       
        def cosine_sim (a1, a2):
            normalized_a1 = tf.nn.l2_normalize (a1, 1)        
            normalized_a2 = tf.nn.l2_normalize (a2, 1)
            d1 = tf.losses.cosine_distance (normalized_a1, normalized_a2, 0)
            #d2 = tf.reduce_sum (tf.multiply (normalized_a1, normalized_a2))
            return 1-d1 #1-d1 # d2

        def cosine_dist_1tensor (a):
            normalized_a = tf.nn.l2_normalize (a, 1)        
            prod = tf.matmul (normalized_a, normalized_a, adjoint_b=True)
            dist = 1 - prod
            return tf.reduce_sum (dist, [0, 1]) / 2.0

        train_data_remain = train_data [:, 0:d_remain]
        train_data_ident = train_data [:, d_remain:]
        (init_w_hid_embed_1_val, init_bias_hid_embed_1_val, init_w_hid_embed_2_val, init_bias_hid_embed_2_val, init_w_out_embed_val, init_bias_out_embed_val, init_w_hid_main_1_val, init_bias_hid_main_1_val, init_w_hid_main_2_val, init_bias_hid_main_2_val, init_w_out_main_val, init_bias_out_main_val) = self.restore_weights_car2vect_regul (train_data_ident, train_car_ident, train_data_remain, train_label, meta_file, ckpt_file) 

        tf.reset_default_graph () # This line is really important, if I restore weights from the pre-trained model, and don't reset the graph to default, the results are weird!
        x_ident = tf.placeholder(tf.float32, [None, d_ident], name="x_ident")
        car_ident = tf.placeholder(tf.float32, [None, 5], name="car_ident")
        x_remain = tf.placeholder(tf.float32, [None, d_remain], name="x_remain")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        print ("build_car2vect_model: d_ident:", d_ident, "d_remain:", d_remain, "d_embed:", d_embed, "no_neuron_embed:", no_neuron_embed, "no_neuron_main:", no_neuron)
        rep_model_code = tf.reshape (car_ident[:, 1], [-1, 1])
        len_data = tf.shape (rep_model_code)[0]

        # He initializer
        # Embeding NN
        he_init = tf.contrib.layers.variance_scaling_initializer ()
        output1 = slim.fully_connected (x_ident, no_neuron_embed, scope='hidden_embed1', activation_fn=tf.nn.relu, weights_initializer=tf.constant_initializer (init_w_hid_embed_1_val), biases_initializer=tf.constant_initializer (init_bias_hid_embed_1_val), weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta))
        output1_ = slim.dropout (output1, self.dropout, scope='dropout1')

        output2 = slim.fully_connected (output1_, no_neuron_embed, scope='hidden_embed2', activation_fn=tf.nn.relu, weights_initializer=tf.constant_initializer (init_w_hid_embed_2_val), biases_initializer=tf.constant_initializer (init_bias_hid_embed_2_val), weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta)) #None) #
        output2_ = slim.dropout (output2, self.dropout, scope='dropout2')

        x_embed = slim.fully_connected (output2_, d_embed, scope='output_embed', activation_fn=None, weights_initializer=tf.constant_initializer (init_w_out_embed_val), biases_initializer=tf.constant_initializer (init_bias_out_embed_val), weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta)) # 3-dimension of embeding NN

        input3 = tf.concat ([x_remain, x_embed], 1)

        # Main NN
        output3 = slim.fully_connected(input3, no_neuron, scope='hidden_main1', activation_fn=tf.nn.relu, weights_initializer=tf.constant_initializer (init_w_hid_main_1_val), biases_initializer=tf.constant_initializer (init_bias_hid_main_1_val), weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta))
        output3_ = slim.dropout (output3, self.dropout, scope='dropout3')

        output4 = slim.fully_connected(output3_, no_neuron, scope='hidden_main2', activation_fn=tf.nn.relu, weights_initializer=tf.constant_initializer (init_w_hid_main_2_val), biases_initializer=tf.constant_initializer (init_bias_hid_main_2_val), weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta))
        output4_ = slim.dropout (output4, self.dropout, scope='dropout4')

        prediction = slim.fully_connected(output4_, 1, scope='output_main', activation_fn=None, weights_initializer=tf.constant_initializer (init_w_out_main_val), biases_initializer=tf.constant_initializer (init_bias_out_main_val), weights_regularizer = tf.contrib.layers.l2_regularizer(self.beta)) 
        tf.identity (prediction, name="prediction")
        
        ####################################
        # Regularization
        ###################################
        regul1 = tf.reduce_sum (tf.get_collection (tf.GraphKeys.REGULARIZATION_LOSSES))

        # Cluster
        x_embed_add_rep = tf.concat ([rep_model_code, x_embed], 1)
        # Sort by rep_model_code
        x_embed_add_rep_sorted = tf.gather (x_embed_add_rep, tf.nn.top_k (x_embed_add_rep [:, 0], k=len_data).indices) 
        rep_model_unique, idx, count = tf.unique_with_counts (x_embed_add_rep_sorted[:, 0])

        # Compute the centroid vectors of every vector in each cluster (rep_model)
        x_embed_mean = tf.segment_mean (x_embed_add_rep_sorted, idx) # the 1st column is rep_model_code, the 3 other columns are x_embed_mean
        # Remove 1st column
        x_embed_mean = x_embed_mean [:, 1:]

        # Normalize all of the centroid vector
        norm = tf.norm (x_embed_mean, 1)
        x_embed_mean_normed = x_embed_mean #/ norm

        # Expand to a vector, compatible with the index of x_embed_sorted
        centroid_normed = tf.gather (x_embed_mean_normed, idx)

        # Remove 1st column
        x_embed_sorted = x_embed_add_rep_sorted [:, 1:]

        # The similiraty of every point with its centroid is added cummulatively 
        #regul_gather = tf.reduce_sum (cosine_sim (x_embed_sorted, centroid)) # want to keep this value large
        # The distance of every point with its centroid is added cummulatively
        regul_gather = tf.sqrt (tf.reduce_sum ((x_embed_sorted - centroid_normed) ** 2)) # want to keep this value small 

        # The distance between every centroid is added cummulatively 
        regul_spread = cosine_dist_1tensor (x_embed_mean_normed) # want to keep this value large

        regul_out = self.gama * regul1 + self.alpha * regul_gather - self.beta * regul_spread #+ self.beta * 1 / regul_spread
        #def f1(): return tf.constant (1.)
        #def f2(): return regul_out
        #regul_out = tf.cond (tf.greater (tf.abs (regul_out), 1), f1, f2)
        ####################################

        return x_ident, x_remain, Y, x_embed, prediction, phase_train, car_ident, regul1, regul_gather, regul_spread, regul_out #, rep_model_code, centroid, x_embed_add_rep_sorted, regul_gather, regul_spread, regul_out

    def restore_model_car2vect (self, sess, x_ident_, x_remain_, label, meta_file, ckpt_file, train_flag):
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
            # Embedding NN
            w_hid_embed_1 = graph.get_tensor_by_name ("hidden_embed1/weights:0")
            bias_hid_embed_1 = graph.get_tensor_by_name ("hidden_embed1/biases:0")

            w_out_embed = graph.get_tensor_by_name ("output_embed/weights:0")
            bias_out_embed = graph.get_tensor_by_name ("output_embed/biases:0")

            # Main NN
            w_hid_main_1 = graph.get_tensor_by_name ("hidden_main1/weights:0")
            bias_hid_main_1 = graph.get_tensor_by_name ("hidden_main1/biases:0")

            w_out_main = graph.get_tensor_by_name ("output_main/weights:0")
            bias_out_main = graph.get_tensor_by_name ("output_main/biases:0")

            # Feed data
            #(w_hid_embed_1_val, bias_hid_embed_1_val, w_out_embed_val, bias_out_embed_val, w_hid_main_1_val, bias_hid_main_1_val, w_out_main_val, bias_out_main_val) 
            return sess.run ([w_hid_embed_1, bias_hid_embed_1, w_out_embed, bias_out_embed, w_hid_main_1, bias_hid_main_1, w_out_main, bias_out_main], feed_dict)

    def restore_weights_car2vect_regul (self, x_ident_, car_ident_, x_remain_, label, meta_file, ckpt_file):
        with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run (init)
            # Load meta graph and restore all variables values
            saver = tf.train.import_meta_graph (meta_file)
            saver.restore (sess, ckpt_file)

            # Access and create placeholder variables, and create feedict to feed data
            graph = tf.get_default_graph ()
            x_ident = graph.get_tensor_by_name ("x_ident:0")
            car_ident = graph.get_tensor_by_name ("car_ident:0")
            x_remain = graph.get_tensor_by_name ("x_remain:0")
            Y = graph.get_tensor_by_name ("Y:0")
            feed_dict = {x_ident:x_ident_, car_ident:car_ident_, x_remain:x_remain_, Y:label}

            # Now, access the weights of the pre-trained model
            # Embedding NN
            w_hid_embed_1 = graph.get_tensor_by_name ("hidden_embed1/weights:0")
            bias_hid_embed_1 = graph.get_tensor_by_name ("hidden_embed1/biases:0")

            w_hid_embed_2 = graph.get_tensor_by_name ("hidden_embed2/weights:0")
            bias_hid_embed_2 = graph.get_tensor_by_name ("hidden_embed2/biases:0")

            w_out_embed = graph.get_tensor_by_name ("output_embed/weights:0")
            bias_out_embed = graph.get_tensor_by_name ("output_embed/biases:0")

            # Main NN
            w_hid_main_1 = graph.get_tensor_by_name ("hidden_main1/weights:0")
            bias_hid_main_1 = graph.get_tensor_by_name ("hidden_main1/biases:0")

            w_hid_main_2 = graph.get_tensor_by_name ("hidden_main2/weights:0")
            bias_hid_main_2 = graph.get_tensor_by_name ("hidden_main2/biases:0")

            w_out_main = graph.get_tensor_by_name ("output_main/weights:0")
            bias_out_main = graph.get_tensor_by_name ("output_main/biases:0")

            # Feed data
            return sess.run ([w_hid_embed_1, bias_hid_embed_1, w_hid_embed_2, bias_hid_embed_2, w_out_embed, bias_out_embed, w_hid_main_1, bias_hid_main_1, w_hid_main_2, bias_hid_main_2, w_out_main, bias_out_main], feed_dict)

    

    def car2vect(self, train_data, train_label, test_data, test_label, total_car_ident, total_act_adv_date, total_sale_date, d_ident, d_embed, d_remain, no_neuron, no_neuron_embed, loss_func, model_path, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, retrain): # Used for 1train-1test, add total_act_adv_date, sale_date
    #def car2vect(self, train_data, train_label, test_data, test_label, total_car_ident, d_ident, d_embed, d_remain, no_neuron, no_neuron_embed, loss_func, model_path, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, retrain): # Used for 1train-1test
    #def car2vect(self, train_data, train_label, test_data, test_label, test_car_ident, d_ident, d_embed, d_remain, no_neuron, no_neuron_embed, loss_func, model_path, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, retrain): # Used for 1train-1test
    #def car2vect(self, train_data, train_label, test_data, test_label, test_car_ident, model_path, d_ident, d_embed, d_remain, x_ident, x_remain, Y, x_embed, prediction, fold): # used for Cross-validation 

        #building car embedding model
        len_train = len(train_data)
        len_test  = len(test_data)
        if len_train > 0:
            d_data = train_data.shape[1]

        train_car_ident = total_car_ident [:len_train, :] 
        test_car_ident = total_car_ident [len_train:, :] 
        test_act_adv_date = total_act_adv_date [len_train:, :] 
        test_sale_date = total_sale_date [len_train:, :] 
        test_data_remain = test_data [:, 0:d_remain]
        test_data_ident = test_data [:, d_remain:]


        if using_CV_flag == 0:
            if retrain == -1:
                print ("=====Restore the trained model...")
                # Apply the trained model onto the test data
                pre_model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}x1_{5}x1_total_set_{6}".format (2, "new", self.model_name, self.label, self.no_neuron_embed, self.no_neuron, 49)
                meta_file = pre_model_path + ".meta"
                ckpt_file = pre_model_path 
                
                if self.test_uncertainty == True:
                    ##### Test uncertainty
                    N = tqdm (range (50))
                    np_pred_y = np.zeros ((test_label.shape[0],0))
                    for i in N:
                        #predicted_y, sum_se_val, sum_ae_val, sum_rel_err_val, sum_smape_val, arr_rel_err_val = sess.run([prediction, sum_se, sum_ae, sum_rel_err, sum_smape, arr_rel_err], feed_dict)
                        (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val, test_arr_relative_err) = self.batch_computation_car2vect (5, test_data, test_label, d_ident, d_remain, meta_file, ckpt_file, 1)
                        print ("---iter:", i, test_relative_err_val)
                        np_pred_y = np.concatenate ((np_pred_y, predicted_test_label), axis=1)

                    predicted_test_label = np_pred_y.mean (axis=1).reshape (-1, 1)
                    uncertainty = np_pred_y.var (axis=1).reshape (-1, 1)
                    std = np_pred_y.std (axis=1).reshape (-1, 1)
                    print ("Label:", test_label[:10])
                    print ("Final prediction:", predicted_test_label[:10])
                    print ("Final uncertainty", uncertainty[:10])
                    print ("Final std", std[:10], min (std), max (std))
                    rel_err = get_relative_err (predicted_test_label, test_label)
                    print ("Rel_err:", rel_err)
                    #sys.exit (-1)

                elif self.test_pred_time == True:
                    ##### Test prediction time
                    list_batch_size = [64]#1, 16, 64, 256, 512, 4096, 8192]
                    len_list = len (list_batch_size)
                    for batch_size in tqdm (list_batch_size, total=len_list):
                        print ("batch_size:", batch_size)
                        n_iter = len_test / batch_size
                        s_time = time.time()
                        (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val, test_arr_relative_err) = self.batch_computation_car2vect (n_iter, test_data, test_label, d_ident, d_remain, meta_file, ckpt_file, 2)
                        pred_time = time.time() - s_time
                        print ("pred_time:", pred_time)
                        line = np.zeros ((0, 3))
                        content = np.array ([use_gpu, args.batch_size, pred_time]).reshape (1, 3)
                        line = np.concatenate ((line,content), axis = 0)
                        with open ("./test_pred_time_{0}.txt".format (use_gpu), "ab") as file:
                            np.savetxt (file, line, fmt="%s\t%s\t%s\t%s")
                    sys.exit (-1)
                else:
                    (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val, test_arr_relative_err) = self.batch_computation_car2vect (5, test_data, test_label, d_ident, d_remain, meta_file, ckpt_file, 1)

                # Save the identification and results into a file
                result = np.concatenate ((test_car_ident, test_act_adv_date, test_sale_date, test_label, predicted_test_label, std),axis=1)
                np.savetxt (x_ident_file_name + "_restored", result, fmt="%d\t%d\t%d\t%d\t%s\t%s\t%d\t%d\t%.2f\t%.2f")  

                return 0 
                
            elif retrain == 0:
                x_ident, x_remain, Y, x_embed, prediction, phase_train = self.build_car2vect_model(no_neuron, no_neuron_embed, d_ident, d_embed, d_remain)
                start_time = time.time()
                #x_ident, x_remain, Y, x_embed, prediction, phase_train, car_ident, regul1, regul_gather, regul_spread, regul= self.build_car2vect_regul_model(no_neuron, no_neuron_embed, d_ident, d_embed, d_remain)
                print ("========== Time for building the new model:", time.time () - start_time)

                """init = tf.global_variables_initializer ()
                sess = tf.InteractiveSession ()
                sess.run (init)

                test_data_remain = test_data [:, 0:d_remain]
                test_data_ident = test_data [:, d_remain:]
                len_train = len(train_data)
                test_car_ident = total_car_ident [len_train:, :] 

                regul_val, regul_gather2_val, regul_spread_val, x_embed_val, x_embed_add_rep_sorted_val = sess.run([regul, regul_gather2, regul_spread, x_embed, x_embed_add_rep_sorted], feed_dict={car_ident:test_car_ident, x_ident: test_data_ident, x_remain: test_data_remain, Y: test_label, phase_train: False})
                sess.close ()
                sys.exit (-1)"""

            ###########
            else:
                self.learning_rate /= 10
                print ("=======new learning_rate:", self.learning_rate)
                #pre_model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect_regul/regressor1/full_{0}_{1}_car2vect_{2}_{3}_total_set".format (self.model_name, self.label, self.no_neuron_embed, self.no_neuron)
                #pre_model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect/regressor1/full_{0}_{1}_car2vect_{2}_{3}_total_set".format (self.model_name, self.label, self.no_neuron_embed, self.no_neuron)
                pre_model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}x1_{5}x1_total_set_{6}".format (2, "new", self.model_name, self.label, self.no_neuron_embed, self.no_neuron, 27)
                meta_file = pre_model_path + ".meta"
                ckpt_file = pre_model_path 
                #x_ident, x_remain, Y, x_embed, prediction, phase_train, car_ident, regul1, regul_gather, regul_spread, regul = self.build_car2vect_regul_model_retrained (no_neuron=no_neuron, no_neuron_embed=no_neuron_embed, d_ident=d_ident, d_embed=d_embed, d_remain=d_remain, train_data=train_data, train_label=train_label, train_car_ident=train_car_ident, meta_file=meta_file, ckpt_file=ckpt_file)
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

        print ("==test car ident", test_car_ident.shape)
        print ("==test act_adv_date", test_act_adv_date.shape)
        print ("==test test_sale_date", test_sale_date.shape)
        identification = np.concatenate ((test_car_ident, test_act_adv_date, test_sale_date),axis=1)
        np.savetxt (x_ident_file_name_, identification, fmt="%d\t%d\t%d\t%d\t%s\t%s\t%d")  

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

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step) # + regul
    
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

            train_set = np.concatenate ((train_data, train_label, train_car_ident), axis = 1)
            train_set_shuffled = np.random.permutation(train_set)
            train_data_remain_shuffled = train_set_shuffled [:, 0:d_remain]
            train_data_ident_shuffled = train_set_shuffled [:, d_remain:d_data]
            train_label_shuffled = train_set_shuffled [:, d_data:d_data+1]
            train_car_ident_shuffled = train_set_shuffled [:, d_data+1:]

            if self.scale_label == 1:
                scaler = MinMaxScaler(feature_range=(self.min_price, self.max_price))
                scaler.fit (train_label_shuffled)
                train_label_shuffled_scaled = scaler.transform (train_label_shuffled)
            else:
                train_label_shuffled_scaled = train_label_shuffled

            print ("len train_data_ident:", train_data_ident_shuffled.shape)
            print ("len train_data_remain:", train_data_remain_shuffled.shape)
            print ("len train_label:", train_label_shuffled.shape)
            print ("len test_data:", test_data.shape)

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
                    batch_car_ident = train_car_ident_shuffled [start_index : end_index]
                    batch_x_remain = train_data_remain_shuffled [start_index : end_index]
                    batch_y = train_label_shuffled_scaled [start_index : end_index]

                    start_index = end_index

                    _, train_sum_se_val, train_sum_ae_val, train_sum_relative_err_val, train_sum_smape_val = sess.run([optimizer, sum_se, sum_ae, sum_relative_err, sum_smape], feed_dict={x_ident: batch_x_ident, x_remain: batch_x_remain, Y: batch_y, phase_train: True})
                    #_, train_sum_se_val, train_sum_ae_val, train_sum_relative_err_val, train_sum_smape_val = sess.run([optimizer, sum_se, sum_ae, sum_relative_err, sum_smape], feed_dict={x_ident: batch_x_ident, car_ident: batch_car_ident, x_remain: batch_x_remain, Y: batch_y, phase_train: True})
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

                """#################################################
                # If we use 2 hidden layers, each has >= 10000 units -> resource exhausted, then we should divide it into batches and test on seperate one, and then calculate the average.
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
                epoch_test_smape_val = total_smape/len_test
                #################################################"""

                # Non-regularization
                predicted_y, x_embed_val, epoch_test_rmse_val, epoch_test_mae_val, epoch_test_relative_err_val, epoch_test_smape_val = sess.run([prediction, x_embed, rmse, mae, relative_err, smape], feed_dict={x_ident: test_data_ident, x_remain: test_data_remain, Y: test_label_scaled, phase_train: False})

                # Regularization
                #predicted_y, x_embed_val, epoch_test_rmse_val, epoch_test_mae_val, epoch_test_relative_err_val, epoch_test_smape_val, epoch_regul1_val, epoch_regul_gather_val, epoch_regul_spread_val, epoch_regul_val= sess.run([prediction, x_embed, rmse, mae, relative_err, smape, regul1, regul_gather, regul_spread, regul], feed_dict={x_ident: test_data_ident, car_ident: test_car_ident, x_remain: test_data_remain, Y: test_label_scaled, phase_train: False})

                
                #print ("test: (regul1, regul_gather, regul_spread, regul)", np.c_[epoch_regul1_val, epoch_regul_gather_val, epoch_regul_spread_val, epoch_regul_val])
                print ("test: rmse", epoch_test_rmse_val)
                print ("test: mae", epoch_test_mae_val)
                print ("test: relative_err", epoch_test_relative_err_val)
                print ("test: smape", epoch_test_smape_val)
                print ("(truth, prediction):", np.c_[test_label[:10], predicted_y[:10]])

                epoch_list.append (epoch)
                train_rmse_list.append (epoch_train_rmse_val)
                train_mae_list.append (epoch_train_mae_val)
                train_rel_err_list.append (epoch_train_relative_err_val)
                train_smape_list.append (epoch_train_smape_val)
                rmse_list.append (epoch_test_rmse_val)
                mae_list.append (epoch_test_mae_val)
                rel_err_list.append (epoch_test_relative_err_val)
                smape_list.append (epoch_test_smape_val)

                line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
                line['truth'] = test_label.reshape (test_label.shape[0])
                line['pred'] = predicted_y.reshape (predicted_y.shape[0])

                # Save predicted label and determine the best epoch
                if loss_func == "rel_err":
                    threshold_err = 6 #7 #6.5 #9.3 #8.5 #
                    epoch_test_err_val = epoch_test_relative_err_val
                elif loss_func == "mae":
                    threshold_err = 15
                    epoch_test_err_val = epoch_test_mae_val
                elif loss_func == "mse":
                    threshold_err = 20 #140
                    epoch_test_err_val = epoch_test_rmse_val

                #if (epoch + 1) % 1 == 0:
                if (epoch_test_err_val < threshold_err) or (epoch == self.epoch - 1):
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
                train_data_ident_shuffled = train_set_shuffled [:, d_remain:d_data]
                train_label_shuffled = train_set_shuffled [:, d_data:d_data+1]
                train_car_ident_shuffled = train_set_shuffled [:, d_data+1:]

                if self.scale_label == 1:
                    train_label_shuffled_scaled = scaler.transform (train_label_shuffled)
                else:
                    train_label_shuffled_scaled = train_label_shuffled


            print ("Training finished!")
            stop_time = time.time()
            print ("Training time (s):", stop_time - start_time)
            
            print('test last epoch rel_err: {:.3f}'.format(epoch_test_relative_err_val))

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


    def train_car2vect (self, train_data, train_label, total_car_ident, d_ident, d_embed, d_remain, no_neuron, no_neuron_embed, loss_func, model_path): 
    #def train_car2vect (self, train_data, train_label, d_ident, d_embed, d_remain, no_neuron, no_neuron_embed, loss_func, model_path): 
        #building car embedding model
        x_ident, x_remain, Y, x_embed, prediction, phase_train = self.build_car2vect_model(no_neuron, no_neuron_embed, d_ident, d_embed, d_remain)
        #x_ident, x_remain, Y, x_embed, prediction, phase_train, car_ident, regul1, regul_gather, regul_spread, regul= self.build_car2vect_regul_model(no_neuron, no_neuron_embed, d_ident, d_embed, d_remain)

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

            len_train = len(train_data)
            d_data = train_data.shape[1]
            train_car_ident = total_car_ident [:len_train, :]


            train_set = np.concatenate ((train_data, train_label, train_car_ident), axis = 1)
            train_set_shuffled = np.random.permutation(train_set)
            train_data_remain_shuffled = train_set_shuffled [:, 0:d_remain]
            train_data_ident_shuffled = train_set_shuffled [:, d_remain:train_data.shape[1]]
            train_label_shuffled = train_set_shuffled [:, d_data:d_data+1]
            train_car_ident_shuffled = train_set_shuffled [:, d_data+1:]

            print ("len train_data_ident:", train_data_ident_shuffled.shape)
            print ("len train_data_remain:", train_data_remain_shuffled.shape)
            print ("len train_label:", train_label_shuffled.shape)

            len_train = len(train_data)
            train_total_batch = int (np.ceil (float (len_train)/self.batch_size))

            epoch_list = [] 
            train_rel_err_list = [] 
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
                    batch_car_ident = train_car_ident_shuffled [start_index : end_index]
                    batch_x_remain = train_data_remain_shuffled [start_index : end_index]
                    batch_y = train_label_shuffled [start_index : end_index]

                    start_index = end_index

                    _, train_sum_se_val, train_sum_ae_val, train_sum_relative_err_val, train_sum_smape_val = sess.run([optimizer, sum_se, sum_ae, sum_relative_err, sum_smape], feed_dict={x_ident: batch_x_ident, x_remain: batch_x_remain, Y: batch_y, phase_train: True})
                    #_, train_sum_se_val, train_sum_ae_val, train_sum_relative_err_val, train_sum_smape_val = sess.run([optimizer, sum_se, sum_ae, sum_relative_err, sum_smape], feed_dict={x_ident: batch_x_ident, car_ident: batch_car_ident, x_remain: batch_x_remain, Y: batch_y, phase_train: True})
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
                if epoch == self.epoch-1 or (epoch + 1)%10 == 0: 
                    ckpt_file = model_path + "_" + str (epoch)
                    save_path = saver.save (sess, ckpt_file)  
                    print('Model saved in file: %s' % save_path)

                # Training data permutation
                train_set_shuffled = np.random.permutation(train_set)
                train_data_remain_shuffled = train_set_shuffled [:, 0:d_remain]
                train_data_ident_shuffled = train_set_shuffled [:, d_remain:d_data]
                train_label_shuffled = train_set_shuffled [:, d_data:d_data+1]
                train_car_ident_shuffled = train_set_shuffled [:, d_data+1:]


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
            (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val) = self.restore_model_NN_baseline (train_data, train_label_copy, meta_file, ckpt_file, train_flag=0)
            (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_NN_baseline (test_data, test_label_copy, meta_file, ckpt_file, train_flag=0)
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
            (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val, train_arr_relative_err) = self.batch_computation_car2vect (5, train_data, train_label_copy, d_ident, d_remain, meta_file, ckpt_file)

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
            print ("(truth, prediction): regressor {0}\n{1}".format (i, np.c_[test_label[:10], predicted_test_label[:10]]))
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

    def remove_outliers_total_set (self, total_data, total_label, total_car_ident_code, act_ad_date, sale_date, total_rel_err, dataset_size, removal_percent):
        """
            Purpose:
            - Remove outliers of the total dataset: the data points in the training set with the corresponding relative error in top (removal_percent)%
            - Then, sort the dataset by actual_advertising_date
            Note:
            - If the array total_car_ident_code is empty -> baseline method
            - Else -> car2vect
        """
        # Save and load file take a lot of times, much more than the common way
        #stored_np_arr_file = "./Dataframe/[{0}]total_numpy_array_after_remove_outliers.h5_{1}".format (dataset_size, removal_percent)
        #removed_np_arr_file = "./Dataframe/[{0}]removed_numpy_array_after_remove_outliers.h5_{1}".format (dataset_size, removal_percent)
        #stored_np_arr_file = "./Dataframe/[{0}]total_numpy_array_after_remove_outliers.h5_{1}".format (dataset_size, removal_percent)
        #key = "df"
        #if os.path.isfile (stored_np_arr_file + ".npy") == False:# or os.path.isfile (removed_np_arr_file) == False:

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
        print ("rm_idx:", rm_idx[:5])
        new_dataset = dataset [idx]

        # Sort by adv_date
        new_dataset = new_dataset [new_dataset[:, -3].argsort()]

        # The new dataset will store sale_date at the end of the matrix
        new_dataset = new_dataset[:, :-1] 

        # Save the dataset into a binary file
        #np.save (stored_np_arr_file, new_dataset)
        # Convert numpy array to dataframe format and save the dataset into a pytable file (compressed)
        #df_dataset = pd.DataFrame (new_dataset)
        #df_dataset.to_hdf (stored_np_arr_file, key)       

        return (new_dataset, rm_idx)

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

    def get_features_importance_baseline_NN (self, train_data, train_label, test_data, test_label, model_path, features, l_feature):
        #meta_file = model_path + "_4.meta"
        #ckpt_file = model_path + "_4" 
        meta_file = "../checkpoint/baseline/test4.meta" 
        ckpt_file = "../checkpoint/baseline/test4" 
        # Train the model based on the train set
        # TODO: Replace this step by using the best model trained 
        #self.train_baseline (train_data, train_label, self.no_neuron, self.no_hidden_layer, self.loss_func, ckpt_file)

        #sys.exit (-1)

        # Restore the model
        #predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val = self.restore_model_NN_baseline (train_data, train_label, meta_file, ckpt_file, train_flag=0)
        (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val, train_arr_relative_err) = self.batch_computation_baseline (5, train_data, train_label, meta_file, ckpt_file)
        print ("Train: ", train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val)

        # Create the list of permutation data
        list_test_data, idx = self.get_permutation_dataset_list (l_feature, test_data)
        #list_test_data = self.get_permutation_dataset_list (l_feature, test_data)
        test_label = test_label [idx]

        len_list_test = len (list_test_data)
        df_importance_score = pd.DataFrame (columns=["feature", "rmse", "mae", "rel_err", "smape"])
        for i in range (len_list_test):
            print ("===Test set:", i)
            #print (list_test_data[i].shape, test_label.shape)
            #sys.exit (-1)
            _, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val = self.restore_model_NN_baseline (list_test_data[i], test_label, meta_file, ckpt_file, train_flag=0)
            print (test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val)
            if i == 0:
                init_rmse_err = test_rmse_val
                init_mae_err = test_mae_val
                init_rel_err = test_relative_err_val
                init_smape_err = test_smape_val
            else:
                df_importance_score.loc[len (df_importance_score)] = [features[i-1], test_rmse_val - init_rmse_err, test_mae_val - init_mae_err, test_relative_err_val - init_rel_err, test_smape_val - init_smape_err]

        print ("\n\n ===============")
        np.savetxt ("./importance_score_baseline.txt", df_importance_score, fmt="%s\t%.2f\t%.2f\t%.2f\t%.2f")

    def test_affect_price_sales_duration_baseline (self, train_data, train_label, test_data, test_label, model_path, y_predict_file_name):
        meta_file = model_path + "4.meta"# + "_49.meta"
        ckpt_file = model_path + "4" 
        # Train the model based on the train set
        # TODO: Replace this step by using the best model trained 
        #self.train_car2vect(train_data, train_label, total_car_ident, d_ident, self.d_embed, d_remain, self.no_neuron, self.no_neuron_embed, self.loss_func, ckpt_file)

        (predicted_test_label, _, _, _, _, _) = self.batch_computation_baseline (5, test_data, test_label, meta_file, ckpt_file)

        line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
        line['truth'] = test_label.reshape (test_label.shape[0])
        line['pred'] = predicted_test_label.reshape (predicted_test_label.shape[0])
        
        np.savetxt (y_predict_file_name, line, fmt="%.2f\t%.2f")

    def test_affect_price_sales_duration_car2vec (self, train_data, train_label, test_data, test_label, d_ident, d_remain, model_path, y_predict_file_name):
        meta_file = model_path + "4.meta"# + "_49.meta"
        ckpt_file = model_path + "4" 
        # Train the model based on the train set
        # TODO: Replace this step by using the best model trained 
        #self.train_car2vect(train_data, train_label, total_car_ident, d_ident, self.d_embed, d_remain, self.no_neuron, self.no_neuron_embed, self.loss_func, ckpt_file)

        (predicted_test_label, _, _, _, _, train_arr_relative_err) = self.batch_computation_car2vect (5, test_data, test_label, d_ident, d_remain, meta_file, ckpt_file)

        line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
        line['truth'] = test_label.reshape (test_label.shape[0])
        line['pred'] = predicted_test_label.reshape (predicted_test_label.shape[0])
        
        np.savetxt (y_predict_file_name, line, fmt="%.2f\t%.2f")

    def decide_when_sell_car (self, list_test_data, test_label, d_ident, d_remain, model_path):
        meta_file = model_path + "_49.meta"
        ckpt_file = model_path + "_49" 

        len_list_test = len (list_test_data)
        df = pd.DataFrame (columns=list (range (len_list_test)))
        for i in range (len_list_test):
            print ("=== Funtion: decide_when_sell_car", "test:", i)
            (predicted_test_label, _, _, _, _, train_arr_relative_err) = self.batch_computation_car2vect (5, list_test_data[i], test_label, d_ident, d_remain, meta_file, ckpt_file)
            df[i] = predicted_test_label.reshape (len (predicted_test_label)).tolist ()

        df.to_excel ("./result_decide_when_sell_car.xlsx", index=False)
        

    def get_features_importance_car2vect (self, train_data, train_label, test_data, test_label, total_car_ident, d_ident, d_remain, model_path, features, l_feature):
        meta_file = model_path + "_9.meta"# 
        ckpt_file = model_path + "_9" 
        # Train the model based on the train set
        # TODO: Replace this step by using the best model trained 
        self.train_car2vect(train_data, train_label, total_car_ident, d_ident, self.d_embed, d_remain, self.no_neuron, self.no_neuron_embed, self.loss_func, model_path)
        print (len (features), len (l_feature))

        #sys.exit (-1)

        # Restore the model
        (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val, train_arr_relative_err) = self.batch_computation_car2vect (5, train_data, train_label, d_ident, d_remain, meta_file, ckpt_file)
        print ("Train: ", train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val)

        # Create the list of permutation data
        list_test_data, idx = self.get_permutation_dataset_list (l_feature, test_data)
        #list_test_data = self.get_permutation_dataset_list (l_feature, test_data)
        test_label = test_label [idx]

        len_list_test = len (list_test_data)
        df_importance_score = pd.DataFrame (columns=["feature", "rmse", "mae", "rel_err", "smape"])
        for i in range (len_list_test):
            print ("===Test set:", i)
            test_data_remain = list_test_data[i][:, 0:d_remain]
            test_data_ident  = list_test_data[i][:, d_remain:]

            _, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val = self.restore_model_car2vect (test_data_ident, test_data_remain, test_label, meta_file, ckpt_file, train_flag=0)
            print (test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val)
            if i == 0:
                init_rmse_err = test_rmse_val
                init_mae_err = test_mae_val
                init_rel_err = test_relative_err_val
                init_smape_err = test_smape_val
            else:
                df_importance_score.loc[len (df_importance_score)] = [features[i-1], test_rmse_val - init_rmse_err, test_mae_val - init_mae_err, test_relative_err_val - init_rel_err, test_smape_val - init_smape_err]

        print ("\n\n ===============")
        np.savetxt ("./importance_score.txt", df_importance_score, fmt="%s\t%.2f\t%.2f\t%.2f\t%.2f")

    def retrain_baseline_from_total_set (self, total_data, total_label, total_car_ident_code, total_act_adv_date, total_sale_date, y_predict_file_name, x_ident_file_name, mean_error_file_name, dataset_size, removal_percent, ensemble_flag, l_feature, features):
        """
            - Purpose: 
                + Train the model baseline with the whole dataset, and then remove the data points with removal_percent highest relative error.
                + Sort the remaining dataset by total_act_adv_date, and divide it into train and test sets
                + Retrain the model car3vect with the new training data from scratch.
        """
        # First train the model on the original dataset
        os.system ("mkdir -p ../checkpoint/rm_outliers_total_set_NN/baseline/regressor1")
        model_path = "../checkpoint/baseline/"

        meta_file = model_path + "test3.meta" 
        ckpt_file = model_path + "test3" 
        print ("\n\n===========Train total set")
        # If comment the below line, you need to check the checkpoint file in regressor1 (it should be compatible with the dataset) 
        #self.epoch=20
        #self.train_baseline (total_data, total_label, self.no_neuron, self.no_hidden_layer, self.loss_func, ckpt_file)

        print ("After train baseline...")
        

        # Restore the trained model
        # When restore model with the whole dataset, it can cause the error: Resource exhausted 
        # Devide the train set into smaller subsets (Eg. 5 subsets), push them to the model and concatenate the predictions later
        # TODO: change the "model_dir" arg to automatically set the directory
        (predicted_total_label, total_rmse_val, total_mae_val, total_relative_err_val, total_smape_val, total_arr_err) = self.batch_computation_baseline (5, total_data, total_label, meta_file, ckpt_file)
        print ("After restoring baseline model...")
        total_np_arr = np.concatenate ((total_car_ident_code, total_act_adv_date, total_sale_date, total_label, predicted_total_label), axis=1)
        print ("After concatenating...")
        total_df = pd.DataFrame (total_np_arr)
        np.savetxt (y_predict_file_name + "_total_before_remove_outliers", total_df, fmt="%d\t%d\t%d\t%d\t%s\t%.0f\t%d\t%d\t%.2f")

        # Remove outliers from the total dataset based on the relative error from the first train, on the other hand sort the dataset by total_act_adv_date
        stime = time.time()
        if removal_percent > 0:
            new_total_set, rm_idx = self.remove_outliers_total_set (total_data, total_label, total_car_ident_code, total_act_adv_date, total_sale_date, total_arr_err, dataset_size, removal_percent)
            np.savetxt ("./test_rm_idx.txt", rm_idx, fmt="%d")
        else:
            raise ValueError ("Removal perentage need to be larger than 0!")
        print ("Time needed to remove outliers from the dataset: %.3f" % (time.time() - stime))

        # Train the car2vect model based on the new dataset
        # Firstly, create the necessary data for baseline model
        shape1_data = total_data.shape[1]
        new_total_car_ident_code = new_total_set [:, shape1_data+1:shape1_data+6]
        new_total_act_adv_date = new_total_set [:, shape1_data+6:shape1_data+7]
        new_total_sale_date = new_total_set [:, shape1_data+7:]

        len_total_set   = new_total_set.shape[0]    
        data_training_percentage = 0.8 # NOTE: percent
        len_train   = int (0.5 + len_total_set * data_training_percentage)
        new_train_set = new_total_set [:len_train, :]
        new_test_set = new_total_set [len_train:, :]

        # Extract the train set
        new_train_data  = new_train_set [:, :shape1_data] 
        new_train_label = new_train_set [:, shape1_data:shape1_data+1]

        # Extract the test set
        new_test_data  = new_test_set [:, :shape1_data] 
        new_test_label = new_test_set [:, shape1_data:shape1_data+1]

        print ("Old dataset:", total_data.shape, total_label.shape)
        print ("New train dataset:", new_train_data.shape, new_train_label.shape)
        print ("New test dataset:", new_test_data.shape, new_test_label.shape)

        if ensemble_flag == 0:
            tf.reset_default_graph ()
            os.system ("mkdir -p ../checkpoint/rm_outliers_total_set_NN/baseline/regressor2")
            model_path = self.model_dir + "/rm_outliers_total_set_NN/baseline/regressor{0}/{1}_{2}_{3}_baseline_{4}x{5}".format (2, dataset_size, self.model_name, self.label, self.no_neuron, self.no_hidden_layer)
            y_predict_file_name_ = y_predict_file_name + "_2"
            mean_error_file_name_ = mean_error_file_name + "_2"
            x_ident_file_name_ = x_ident_file_name + "_2"
            
            print ("\n\n===========Predictor2")
            self.epoch=50
            best_epoch = self.baseline (train_data=new_train_data, train_label=new_train_label, test_data=new_test_data, test_label=new_test_label, total_car_ident=new_total_car_ident_code, total_act_adv_date=new_total_act_adv_date, total_sale_date=new_total_sale_date, no_neuron=self.no_neuron, no_hidden_layer = self.no_hidden_layer, loss_func=self.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, x_ident_file_name=x_ident_file_name_, mean_error_file_name=mean_error_file_name_)
            print ("Best epoch: ", best_epoch)

        elif ensemble_flag == -1:
            ########################################
            ### Feature importance

            # Get the list of test_data with each feature column is shuffled (the 1st element is the original new_test_data)
            """length = len (l_feature)
            sum_l = 0
            arr_sum_l = []
            list_test_data = [new_test_data]
            for i in range (length-1):
                sum_l += l_feature[i]
                arr_sum_l.append (sum_l)

            X = np.split (new_test_data, indices_or_sections=arr_sum_l, axis=1)
            arr_sum_l.insert (0,0)
            for i in range (length):
                print ("====i:", i)
                test_data_copy = new_test_data.copy ()
                if i < length-1:
                    test_data_copy [:, arr_sum_l [i]:arr_sum_l [i+1]] = np.random.permutation (X[i]) 
                else:
                    test_data_copy [:, arr_sum_l [i]:] = np.random.permutation (X[i]) 
                list_test_data.append (test_data_copy)
            print (len (list_test_data))"""
            tf.reset_default_graph ()
            self.get_features_importance_baseline_NN (new_train_data, new_train_label, new_test_data, new_test_label, model_path, features, l_feature)
            ########################################

    def get_permutation_dataset_list (self, l_feature, test_data):
        # Get the list of test_data with each feature column is shuffled (the 1st element is the original test_data)
        length = len (l_feature)
        sum_l = 0
        arr_sum_l = []
        ########################################################
        ## Only concern the feature importance of specific cars
        print (test_data.shape)
        #print (test_data[:5, :5])
        #sys.exit (-1)
        idx = np.where (test_data[:, 33] == 1) 
        #NOTE:[maping for Price] (0+2=)2-> 1101, (24+2=)26 -> 1132, (45+2=)47 -> 1166 (car class) # 2-only test hyu-kia
        #NOTE:[maping for SD]    (0+2=)2-> 1101, (18+2=)20 -> 1132, (31+2=)33 -> 1166 (car class) # hyu-kia 
        #NOTE:[maping for SD]    (51+0=)51-> 1101, (51+18=)69 -> 1132, (51+31=)82 -> 1166 (car class) # all cars
        test_data = test_data[idx]
        print (test_data.shape)
        #######################################################
        list_test_data = [test_data]
        for i in range (length-1):
            sum_l += l_feature[i]
            arr_sum_l.append (sum_l)

        X = np.split (test_data, indices_or_sections=arr_sum_l, axis=1)
        arr_sum_l.insert (0,0)
        for i in range (length):
            print ("====i:", i)
            test_data_copy = test_data.copy ()
            if i < length-1:
                test_data_copy [:, arr_sum_l [i]:arr_sum_l [i+1]] = np.random.permutation (X[i]) 
            else:
                test_data_copy [:, arr_sum_l [i]:] = np.random.permutation (X[i]) 
            list_test_data.append (test_data_copy)
        return (list_test_data, idx)

    def retrain_car2vect_from_total_set (self, total_data, total_label, total_car_ident_code, total_act_adv_date, total_sale_date, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, removal_percent, ensemble_flag, l_feature, features):# add first_adv_date, sale_date
    #def retrain_car2vect_from_total_set (self, total_data, total_label, total_car_ident_code, total_act_adv_date, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, removal_percent, ensemble_flag, l_feature, features):
        """
            - Purpose: 
                + Train the model car2vect with the whole dataset, and then remove the data points with removal_percent highest relative error.
                + Sort the remaining dataset by total_act_adv_date, and divide it into train and test sets
                + Retrain the model car3vect with the new training data (if the "retrain" flag == 1 -> use the initial weights from the 1st train, 
                otherwise retrain from scratch.
        """
        ##########################
        # First train the model on the original train data (can remove a part of outliers previously)
        os.system ("mkdir -p ../checkpoint/rm_outliers_total_set_NN/car2vect/regressor1")
        model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}_{5}_total_set".format (1, dataset_size, self.model_name, self.label, self.no_neuron_embed, self.no_neuron)

        print ("\n\n===========Train total set")
        # If comment the below line, you need to check the checkpoint file in regressor1 (it should be compatible with the dataset) 
        # Flexible rel_err.
        self.epoch=20
        #self.train_car2vect(train_data=total_data, train_label=total_label, total_car_ident=total_car_ident_code, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func=self.loss_func, model_path=model_path)
        # Only use the below line for Gradient Boosting 
        #self.train_car2vect(train_data=total_data, train_label=total_label, total_car_ident=total_car_ident_code, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func="rel_err", model_path=model_path)

        
        # Restore the trained model
        # When restore model with the whole dataset, it can cause the error: Resource exhausted 
        # Devide the train set into smaller subsets (Eg. 5 subsets), push them to the model and concatenate the predictions later
        # TODO: change the "model_dir" arg to automatically set the directory
        #meta_file = model_path + "_19.meta"
        #ckpt_file = model_path + "_19"
        meta_file = model_path + "_" + str (self.epoch-1) + ".meta"
        ckpt_file = model_path + "_" + str (self.epoch-1)

        (predicted_total_label, total_rmse_val, total_mae_val, total_relative_err_val, total_smape_val, total_arr_relative_err) = self.batch_computation_car2vect (5, total_data, total_label, d_ident, d_remain, meta_file, ckpt_file, 1)
        total_np_arr = np.concatenate ((total_car_ident_code, total_act_adv_date, total_sale_date, total_label, predicted_total_label), axis=1)
        total_df = pd.DataFrame (total_np_arr)
        np.savetxt (y_predict_file_name + "_total_before_remove_outliers", total_df, fmt="%d\t%d\t%d\t%d\t%s\t%.0f\t%d\t%d\t%.2f")

        ############################################################################
        # TRY TO ADD PRED_PRICE TO THE DATASET TO PREDICT SALE_DURATION AFTER REMOVING OUTLIERS
        #print ("Store the dataframe into a hdf file")
        #total_dataset.to_hdf (dataframe_file, key)       
        ############################################################################

        # Remove outliers from the total dataset based on the relative error from the first train, on the other hand sort the dataset by total_act_adv_date
        stime = time.time()
        if removal_percent > 0:
            new_total_set, rm_idx = self.remove_outliers_total_set (total_data, total_label, total_car_ident_code, total_act_adv_date, total_sale_date, total_arr_relative_err, dataset_size, removal_percent)
            np.savetxt ("./test_rm_idx.txt", rm_idx, fmt="%d")
        else:
            raise ValueError ("Removal perentage is 0!")
        print ("Time for remove outliers from dataset: %.3f" % (time.time() - stime))

        #sys.exit (-1)
        # Train the car2vect model based on the new dataset
        # Firstly, create the necessary data for car2vect model
        ################################################
        ## Cach 1
        """shape1_data = total_data.shape[1]
        new_total_data  = new_total_set [:, :shape1_data] 
        new_total_label = new_total_set [:, shape1_data:shape1_data+1]
        new_total_car_ident_code = new_total_set [:, shape1_data+1:]
        
        len_total_set   = new_total_data.shape[0]    
        data_training_percentage = 0.8
        len_train   = int (0.5 + len_total_set * data_training_percentage)
        new_train_data     = new_total_data[:len_train, :]
        new_train_label    = new_total_label[:len_train]
        new_test_data      = new_total_data[len_train:, :]
        new_test_label     = new_total_label[len_train:]
        new_test_car_ident = new_total_car_ident_code[len_train:, :]"""

        ################################################
        ## Cach 2
        shape1_data = total_data.shape[1]
        new_total_car_ident_code = new_total_set [:, shape1_data+1:shape1_data+6]
        new_total_act_adv_date = new_total_set [:, shape1_data+6:shape1_data+7]
        new_total_sale_date = new_total_set [:, shape1_data+7:]

        len_total_set   = new_total_set.shape[0]    
        data_training_percentage = 0.8
        len_train   = int (0.5 + len_total_set * data_training_percentage)
        new_train_set = new_total_set [:len_train, :]
        new_test_set = new_total_set [len_train:, :]

        # Extract train set
        new_train_data  = new_train_set [:, :shape1_data] 
        new_train_label = new_train_set [:, shape1_data:shape1_data+1]

        # Extract test set
        new_test_data  = new_test_set [:, :shape1_data] 
        new_test_label = new_test_set [:, shape1_data:shape1_data+1]
        new_test_car_ident = new_total_car_ident_code[len_train:, :]

        ################################################

        print ("Old dataset:", total_data.shape, total_label.shape)
        print ("New train dataset:", new_train_data.shape, new_train_label.shape)
        print ("New test dataset:", new_test_data.shape, new_test_label.shape)
        print ("New test_car_ident:", new_test_car_ident.shape)

        if ensemble_flag == 0:
            tf.reset_default_graph ()
            os.system ("mkdir -p ../checkpoint/rm_outliers_total_set_NN/car2vect/regressor2")
            model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}x1_{5}x1_total_set".format (2, dataset_size, self.model_name, self.label, self.no_neuron_embed, self.no_neuron)
            #os.system ("mkdir -p ../checkpoint/rm_outliers_total_set_NN/car2vect_regul/regressor2")
            #model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect_regul/regressor{0}/{1}_{2}_{3}_car2vect_{4}_{5}_total_set".format (2, dataset_size, self.model_name, self.label, self.no_neuron_embed, self.no_neuron)
            y_predict_file_name_ = y_predict_file_name + "_2"
            mean_error_file_name_ = mean_error_file_name + "_2"
            x_ident_file_name_ = x_ident_file_name + "_2"
            x_embed_file_name_ = x_embed_file_name + "_2"
            
            print ("\n\n===========Predictor2")
            self.epoch=50
            print (self.epoch)
            best_epoch = self.car2vect (train_data=new_train_data, train_label=new_train_label, test_data=new_test_data, test_label=new_test_label, total_car_ident=new_total_car_ident_code, total_act_adv_date=new_total_act_adv_date, total_sale_date=new_total_sale_date, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func=self.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_, x_ident_file_name=x_ident_file_name_, x_embed_file_name=x_embed_file_name_, retrain=0) # if use retrain=1 -> initialize weights from the previous model, -1 if already have the trained model
            print ("Best epoch: ", best_epoch)

            """#####################
            meta_file = model_path + "_" + str (best_epoch) + ".meta"
            ckpt_file = model_path + "_" + str (best_epoch) 
            test_data_remain = new_test_data [:, 0:d_remain]
            test_data_ident = new_test_data [:, d_remain:]
            (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_car2vect (test_data_ident, test_data_remain, new_test_label, meta_file, ckpt_file, train_flag=0)
            print (predicted_test_label[:5])
            #####################"""

        elif ensemble_flag == 2:
           self.gradient_boosting_NN_car2vect (new_train_data, new_train_label, new_test_data, new_test_label, new_test_car_ident, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, retrain=2) 

        elif ensemble_flag == 5:
            self.bagging_NN_car2vect (new_train_data, new_train_label, new_test_data, new_test_label, new_total_car_ident_code, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, self.sample_ratio, retrain=1)

        elif ensemble_flag == -1:
            ########################################
            ### Feature importance

            tf.reset_default_graph ()
            os.system ("mkdir -p ../checkpoint/rm_outliers_total_set_NN/car2vect/regressor2")
            model_path = self.model_dir + "/rm_outliers_total_set_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}x1_{5}x1_total_set_{6}".format (2, dataset_size, self.model_name, self.label, self.no_neuron_embed, self.no_neuron, 34)
            self.get_features_importance_car2vect (new_train_data, new_train_label, new_test_data, new_test_label, new_total_car_ident_code, d_ident, d_remain, model_path, features, l_feature)
            ########################################

        else:
            raise ValueError ("[retrain_car2vect_from_total_set] This model is not supported!")
        print ("Time for running: retrain_car2vect_from_total_set: %.3f" % (time.time() - stime))

    def predict_sd_with_pred_price (self, train_data_1, train_label_1, train_data_2, test_data_2, test_data, test_label):
        # Problems:
        # - The features used for predicting price and SD are different: price, seller_id, affiliate_id
        # -> Solution: save 2 distinct datasets
        #

        # Train the 1st part to predict price

        # Get the predicted price for the 2nd part

        # Train the 2nd part with joined pred_price to predict sales duration

        pass 

    def batch_computation_baseline  (self, no_batch, train_data, train_label, meta_file, ckpt_file):
        # devide the train set into smaller subsets, push them to the model and concatenate them later
        total_se = 0
        total_ae = 0
        total_relative_err = 0
        total_smape = 0
        start_index = 0
        end_index = 0
        total_predicted_y = np.zeros ((0, 1))
        total_arr_err = np.zeros ((0, 1))
        len_train = len(train_data)
        batch_size = int (len_train / no_batch)
        train_total_batch = int (np.ceil (float (len_train)/batch_size))
        for i in range (train_total_batch):
            if len_train - end_index < batch_size:
                end_index = len_train
            else:
                end_index = (i+1) * batch_size

            batch_x = train_data [start_index : end_index]
            batch_y = train_label [start_index : end_index]

            start_index = end_index

            # TODO: use restore_model_NN_baseline
            (predicted_y, train_sum_se_val, train_sum_ae_val, train_sum_relative_err_val, train_sum_smape_val, arr_se_val, arr_ae_val, arr_rel_err_val) = self.restore_model_NN_baseline (batch_x, batch_y, meta_file, ckpt_file, train_flag=1)
            total_predicted_y = np.concatenate ((total_predicted_y, predicted_y))
            if self.loss_func == "rel_err":
                total_arr_err = np.concatenate ((total_arr_err, arr_rel_err_val))
            elif self.loss_func in ["mse", "mae"]:
                total_arr_err = np.concatenate ((total_arr_err, arr_ae_val))
            else:
               raise ValueError ("Check the loss function name!") 

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
        print (total_arr_err.shape, np.max (total_arr_err), np.min (total_arr_err), np.average (total_arr_err), train_relative_err_val)
        return (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val, total_arr_err)
    
    def batch_computation_car2vect (self, no_batch, data, label, d_ident, d_remain, meta_file, ckpt_file, flag):
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

                (predicted_y, sum_se_val, sum_ae_val, sum_rel_err_val, sum_smape_val, arr_rel_err_val) = self.restore_model_car2vect (sess, batch_x_ident, batch_x_remain, batch_y, meta_file, ckpt_file, flag)
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
        (predicted_train_label, train_rmse_val, train_mae_val, train_relative_err_val, train_smape_val) = self.restore_model_NN_baseline (train_data, train_label_copy, meta_file, ckpt_file, train_flag=0)
        (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_NN_baseline (test_data, test_label_copy, meta_file, ckpt_file, train_flag=0)
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

    def bagging_NN_baseline (self, train_data, train_label, test_data, test_label, total_car_ident, total_act_adv_date, total_sale_date, y_predict_file_name, x_ident_file_name, mean_error_file_name, dataset_size, ratio):
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
            x_ident_file_name_ = x_ident_file_name + "_" + str (i+1)

            # Reset to the default graph, avoid to the error of redefining variables
            tf.reset_default_graph ()

            # Training
            best_epoch =  self.baseline (train_data=X_train, train_label=y_train, test_data=test_data, test_label=test_label, total_car_ident=total_car_ident, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, no_neuron=self.no_neuron, no_hidden_layer=self.no_hidden_layer, loss_func=self.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, x_ident_file_name=x_ident_file_name_, mean_error_file_name=mean_error_file_name_)
            best_epoch = 29
            bash_cmd = "cd ../checkpoint/bagging_NN/baseline/regressor" + str (i+1) + "; mkdir temp_save; rm temp_save/*; cp checkpoint *_" + str (best_epoch) + ".*" + " temp_save; cd ../../../../Code"
            print ("bash_cmd:", bash_cmd)
            os.system (bash_cmd)
            print ("Best epoch: ", best_epoch)
            meta_file = model_path + "_" + str (best_epoch) + ".meta"
            ckpt_file = model_path + "_" + str (best_epoch) 
            (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_NN_baseline (test_data, test_label, meta_file, ckpt_file, train_flag=0)
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

    def bagging_NN_car2vect (self, train_data, train_label, test_data, test_label, total_car_ident, d_ident, d_remain, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, dataset_size, ratio, retrain):
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
            
            best_epoch = self.car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, total_car_ident=total_car_ident, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func=self.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_, x_ident_file_name=x_ident_file_name_, x_embed_file_name=x_embed_file_name_, retrain=0) # if use retrain=1 -> initialize weights from the previous model
            #best_epoch = self.car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, test_car_ident=test_car_ident, d_ident=d_ident, d_embed=self.d_embed, d_remain=d_remain, no_neuron=self.no_neuron, no_neuron_embed=self.no_neuron_embed, loss_func="rel_err", model_path=model_path, y_predict_file_name=y_predict_file_name_, mean_error_file_name=mean_error_file_name_, x_ident_file_name=x_ident_file_name_, x_embed_file_name=x_embed_file_name_, retrain=1)#retrain)
            bash_cmd = "cd ../checkpoint/bagging_NN/car2vect/regressor" + str (i+1) + "; mkdir temp_save; rm temp_save/*; cp checkpoint *_" + str (best_epoch) + ".*" + " temp_save; cd ../../../../Code"
            print ("bash_cmd:", bash_cmd)
            os.system (bash_cmd)
            print ("Best epoch: ", best_epoch)
            meta_file = model_path + "_" + str (best_epoch) + ".meta"
            ckpt_file = model_path + "_" + str (best_epoch) 
            print ("=== meta_file: {0}, ckpt_file: {1}".format (meta_file, ckpt_file))
            (predicted_test_label, test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = self.restore_model_car2vect (test_data_ident, test_data_remain, test_label, meta_file, ckpt_file, train_flag=0)
            list_predicted_test_label.append (predicted_test_label)
            print ("(truth, prediction):", np.c_[test_label[:10], predicted_test_label[:10]])
            print ("=================================")

        print ("label", test_label[:10])
        for i in range (self.num_regressor):
            predicted_test_label = sum (pred_y for pred_y in list_predicted_test_label[:i+1]) / (i+1)
            print ("(truth, prediction): regressor {0}\n{1}".format (i, np.c_[test_label[:10], predicted_test_label[:10]]))
            (test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val) = get_err (predicted_test_label, test_label)
            print ("Err after " + str (i+1)  + " regressors:", test_rmse_val, test_mae_val, test_relative_err_val, test_smape_val)
            line = np.zeros(len (test_label), dtype=[('truth', float), ('pred', float)])
            line['truth'] = test_label.reshape (test_label.shape[0])
            line['pred'] = predicted_test_label.reshape (predicted_test_label.shape[0])
            np.savetxt (y_predict_file_name_ + "_final" + str(i), line, fmt="%.2f\t%.2f")



        
