# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:37:10 2017

@author: quang

CALCULATE PERFORMANCE, PREDICT NEW DATA
"""

from Header_lib import *
from Used_car_regression import *

"""
    EVALUATE MODEL:
     - Get prediction labels (price, sale duration) for new data
     - Get error percentage
"""

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
        
class Test_validation (Dataset, Evaluation):
    
    def __init__ (self):
        pass
    
    # Test scoring function
    def test_built_in_scoring_function (self):
    
        #----------------- TEST LINEAR-------------------------------------
        regr = training.linear_regression (features, output)
        #print (regr.coef_.T.shape)
        score_linear_model = self.get_score_of_model (regr, test_data_matrix_bar, test_label_array)
        print ('***===Score of linear model:')
        print (score_linear_model)
    
        
        #----------------- TEST RIDGE-------------------------------------
        print ('---Score of ridge model with validation set and various penalty:')
        l2_penalty_array = np.logspace (-2, 0, num = 3)#(-9, 5, num = 15)
        len_penalty_array = len (l2_penalty_array)  
        max_score = 0.0
        for i in range (len_penalty_array):
            regr = training.ridge_regression (features, output, l2_penalty = l2_penalty_array[i])
            score_validation_ridge_model = self.get_score_of_model (regr, validation_data_matrix_bar, validation_label_array)
            if score_validation_ridge_model > max_score:
                (max_score, selected_index) = (score_validation_ridge_model, i)
            #print ('---Score of ridge model with penalty ', l2_penalty_array[i], ': ', score_validation_ridge_model)
            print (l2_penalty_array[i], score_validation_ridge_model)
        
        #selected_index = np.argmin(score_validation_ridge_array) 
        print ('***---Max score of ridge model over all penalty: ', max_score, ' with penalty: ', l2_penalty_array[selected_index])
        regr = training.ridge_regression (features, output, l2_penalty = l2_penalty_array[selected_index])
        score_test_ridge = self.get_score_of_model (regr, test_data_matrix_bar, test_label_array)
        print ('***---Score of ridge model with test set: ', score_test_ridge)
    
    
        #----------------- TEST LASSO-------------------------------------
        print ('---Score of lasso model with validation set and various penalty:')
        l2_penalty_array = np.logspace (-2, 0, num = 3)#(-9, 5, num = 15)
        len_penalty_array = len (l2_penalty_array)  
        max_score = 0.0
        for i in range (len_penalty_array):
            regr = training.lasso_regression (features, output, l2_penalty = l2_penalty_array[i])
            score_validation_lasso_model = self.get_score_of_model (regr, validation_data_matrix_bar, validation_label_array)
            if score_validation_lasso_model > max_score:
                (max_score, selected_index) = (score_validation_lasso_model, i)
            #print ('---Score of lasso model with penalty ', l2_penalty_array[i], ': ', score_validation_lasso_model)
            print (l2_penalty_array[i], score_validation_lasso_model)
        
        #selected_index = np.argmin(score_validation_ridge_array) 
        print ('***---Max score of lasso model over all penalty: ', max_score, ' with penalty: ', l2_penalty_array[selected_index])
        regr = training.lasso_regression (features, output, l2_penalty = l2_penalty_array[selected_index])
        score_test_lasso = self.get_score_of_model (regr, test_data_matrix_bar, test_label_array)
        print ('***---Score of lasso model with test data: ', score_test_lasso) 
        
        #----------------- TEST TREE-------------------------------------
        regr = training.tree_GradientBoostingRegressor (features, output, learning_rate = 0.2)
        score_tree = self.get_score_of_model (regr, test_data_matrix_bar, test_label_array)
        print ('***---Score of tree model with learning rate: 0.2:', score_tree)
        
        print ('---Score of Gradient Boosting for Regression model with validation set and various learning rates:')
        learning_rate_array = np.logspace (-2, -0.5, num = 3)#(-2, -0.5, num = 20)
        len_learning_rate_array = len (learning_rate_array)  
        max_score = 0.0
        for i in range (len_learning_rate_array):
            regr = training.tree_GradientBoostingRegressor (features, output, learning_rate = learning_rate_array[i])
            score_validation_tree = self.get_score_of_model (regr, validation_data_matrix_bar, validation_label_array)
        
            if score_validation_tree > max_score:
                (max_score, selected_index) = (score_validation_tree, i)
            #print ('---Score of tree model with learning rate ', learning_rate_array[i], ': ', score_validation_tree)        
            print (learning_rate_array[i], score_validation_tree)        
            
        print ('***+++Max score of tree model over all learning rate: ', max_score, ' with learning rate: ', learning_rate_array[selected_index])
        regr = training.tree_GradientBoostingRegressor (features, output, learning_rate = learning_rate_array[selected_index])    
        score_test_ridge = self.get_score_of_model (regr, test_data_matrix_bar, test_label_array)
        print ('***+++Score of tree model with test data: ', score_test_ridge)
        
     
    #test_built_in_scoring_function ()   
    
    def test_rmse_linear (self):#regr, training_data_matrix_bar, training_label_array, test_data_matrix_bar, test_label_array):
        
        print ("----------------- TEST LINEAR-------------------------------------")
        #stime = time.time()
        
        regr = training.linear_regression (features, output)  
        
        training_predicted_label = self.get_regression_predictions(regr, training_data_matrix_bar)
        training_rmse = self.get_RMSE(training_predicted_label, training_label_array)
        
        test_predicted_label = self.get_regression_predictions(regr, test_data_matrix_bar)   
        test_rmse = self.get_RMSE(test_predicted_label, test_label_array)
        
        print ('***===rmse of linear model in training set:', training_rmse)
        print ('***===rmse of linear model in test set:', test_rmse)
        #print ("Time for Linear regression: %.3f (s)" % (time.time() - stime))
    
    
    def test_rmse_ridge (self):
        
        print ("----------------- TEST RIDGE-------------------------------------")
        #stime = time.time()
        print ('---rmse of ridge model with validation set and various penalty:')
        
        no_penalties = 41
        l2_penalty_array = np.logspace (-20, 20, num = no_penalties)
        len_penalty_array = len (l2_penalty_array)  
        
        final_rmse = 10**6
        
        for i in range (len_penalty_array):
            
            regr = training.ridge_regression (features, output, l2_penalty = l2_penalty_array[i])
            
            training_predicted_label   = self.get_regression_predictions(regr, training_data_matrix_bar)
            validation_predicted_label = self.get_regression_predictions(regr, validation_data_matrix_bar)
            test_predicted_label       = self.get_regression_predictions(regr, test_data_matrix_bar)
  
            training_rmse   = self.get_RMSE(training_predicted_label, training_label_array)            
            validation_rmse = self.get_RMSE(validation_predicted_label, validation_label_array)
            test_rmse       = self.get_RMSE(test_predicted_label, test_label_array)
            
            if validation_rmse < final_rmse:
                (final_rmse, selected_index) = (validation_rmse, i)
                
            #print ('---rmse of ridge model with penalty ', l2_penalty_array[i], ': ', validation_rmse)
            print (l2_penalty_array[i], training_rmse, validation_rmse, test_rmse)
        
        #selected_index = np.argmin(rmse_validation_ridge_array) 
        print ('***---min rmse of ridge model over all penalty: ', final_rmse, ' with penalty: ', l2_penalty_array[selected_index])
        regr = training.ridge_regression (features, output, l2_penalty = l2_penalty_array[selected_index])       
        test_predicted_label = self.get_regression_predictions(regr, test_data_matrix_bar)
        test_rmse = self.get_RMSE(test_predicted_label, test_label_array)
        print ('***---rmse of ridge model with test set: ', test_rmse)
        #print ("Time for ridge regression: %.3f (s)" % (time.time() - stime), "(%d times)" %no_penalties )
        
        
    def test_rmse_lasso (self):
        
        print ("----------------- TEST LASSO-------------------------------------")
        stime = time.time()
        print ('---rmse of lasso model with validation set and various penalty:')
        no_penalties = 41
        l2_penalty_array = np.logspace (-20, 20, num = no_penalties)
        len_penalty_array = len (l2_penalty_array)  
        final_rmse = 10**6
        for i in range (len_penalty_array):
            
            regr = training.lasso_regression (features, output, l2_penalty = l2_penalty_array[i])
            
            training_predicted_label = self.get_regression_predictions(regr, training_data_matrix_bar)
            validation_predicted_label = self.get_regression_predictions(regr, validation_data_matrix_bar)           
            test_predicted_label = self.get_regression_predictions(regr, test_data_matrix_bar)
            
            training_rmse = self.get_RMSE(training_predicted_label, training_label_array)
            validation_rmse = self.get_RMSE(validation_predicted_label, validation_label_array)
            test_rmse = self.get_RMSE(test_predicted_label, test_label_array)
            
            if validation_rmse < final_rmse:
                (final_rmse, selected_index) = (validation_rmse, i)
                
            #print ('---rmse of ridge model with penalty ', l2_penalty_array[i], ': ', validation_rmse)
            print (l2_penalty_array[i], training_rmse, validation_rmse, test_rmse)
        
        #selected_index = np.argmin(rmse_validation_ridge_array) 
        print ('***---min rmse of lasso model over all penalty: ', final_rmse, ' with penalty: ', l2_penalty_array[selected_index])
        regr = training.lasso_regression (features, output, l2_penalty = l2_penalty_array[selected_index])   
        test_predicted_label = self.get_regression_predictions(regr, test_data_matrix_bar)
        test_rmse = self.get_RMSE(test_predicted_label, test_label_array)
        print ('***---rmse of lasso model with test data: ', test_rmse)
        print ("Time for lasso regression: %.3f (s)" % (time.time() - stime), "(%d times)" %no_penalties )
    
    
    def test_rmse_tree (self):
        
        print ("----------------- TEST TREE-------------------------------------")
        stime = time.time()
        regr = training.tree_GradientBoostingRegressor (features, output, learning_rate = 0.2)
        rmse_tree = self.get_RMSE (test_data_matrix_bar, test_label_array)
        print ('***---rmse of tree model with learning rate: 0.2:', rmse_tree)
        
        print ('---rmse of Gradient Boosting for Regression model with validation set and various learning rates:')
        no_penalties = 6
        learning_rate_array = np.logspace (-5, -0.5, num = no_penalties)#(-2, -0.5, num = 20)
        len_learning_rate_array = len (learning_rate_array)  
        final_rmse = 10**6
        for i in range (len_learning_rate_array):
            
            regr = training.lasso_regression (features, output, l2_penalty = l2_penalty_array[i])
            
            training_predicted_label   = self.get_regression_predictions (regr, training_data_matrix_bar)         
            validation_predicted_label = self.get_regression_predictions (regr, validation_data_matrix_bar)           
            test_predicted_label       = self.get_regression_predictions (regr, test_data_matrix_bar)
            
            training_rmse   = self.get_RMSE (training_predicted_label, training_label_array)
            validation_rmse = self.get_RMSE (validation_predicted_label, validation_label_array)
            test_rmse       = self.get_RMSE (test_predicted_label, test_label_array)
            
            if validation_rmse < final_rmse:
                (final_rmse, selected_index) = (validation_rmse, i)
                
            #print ('---rmse of ridge model with penalty ', l2_penalty_array[i], ': ', validation_rmse)
            print (l2_penalty_array[i], training_rmse, validation_rmse)       
            
        print ('***+++min rmse of tree model over all learning rate: ', final_rmse, ' with learning rate: ', learning_rate_array[selected_index])
        regr = training.tree_GradientBoostingRegressor (features, output, learning_rate = learning_rate_array[selected_index])      
        test_predicted_label = self.get_regression_predictions(regr, test_data_matrix_bar)
        test_rmse = self.get_RMSE(test_predicted_label, test_label_array)
        print ('***+++rmse of tree model with test data: ', test_rmse)
        print ("Time for Gradient Boosting for Regression model regression: %.3f (s)" % (time.time() - stime), "%d time" %no_penalties )
        
        
    def test_RMSE_scoring_function (self):
        
        
        self.test_rmse_linear ()
        
        self.test_rmse_ridge ()
        
        self.test_rmse_lasso ()
        
        self.test_rmse_tree ()
        
class Cross_validation (Dataset, Evaluation):
    
    def __init__(self, features, output, list_regression_model, n_splits = 5):
        """
            - Purpose: Initialization
            - Input:
             + features: an array of feature names
             + output: name of the output (label)
             + n_splits: K in k-fold CV
        """
        
        """ NOTE!!!: To run the training faster, we will call the Load_dataset first => have total_dataset variable
            After that, can init the Dataset in this class if it is necessary
        """
        Dataset.__init__(self)
        self.K = n_splits
        self.list_regression_model = list_regression_model
        self.features = features
        self.feature_constraint = feature_constraint
        self.feature_constraint_values = feature_constraint_values
        self.dropout = dropout_h 
        
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
        if output == "price":
            y_total_set = self.get_data_array (self.total_dataset, output)
        elif output == "sale_duration":
            y_total_set = self.get_sale_duration_array (self.total_dataset)
        print ("test:", y_total_set)
        X_train_set = []
        y_train_set = []
        X_test_set = []
        y_test_set = []
        
        kf = KFold(self.K)
        
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
        
        kf = KFold(self.K)
        
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
        
    def regression (self, model, X_train_bar, y_train, l2_penalty = 0, max_depth = 0, max_leaf_nodes = 0, alpha_NN = 0, hidden_layer_sizes = ()): 
        """
            - Purpose: train a model using dataset made by get_data_label()

            - Input:
             + model: "linear", "ridge", "lasso"
             + X_train_bar: the matrix of the expanded training datas
             + y_train: the vector of traning labels
             + l2_penalty: only used for "ridge", and "lasso"

            - Return: regr (regression model)
        """
        
        if model == "linear":
            regr = linear_model.LinearRegression () #fit_intercept = False)
            regr.fit (X_train_bar, y_train)
        
        elif model == "ridge":
            regr = linear_model.Ridge (alpha = l2_penalty, normalize = True)
            regr.fit (X_train_bar, y_train)
        
        elif model == "lasso":
            regr = linear_model.Lasso (alpha = l2_penalty, normalize = True)
            regr.fit (X_train_bar, y_train)
            
        elif model == "basic_tree":
            regr = tree.DecisionTreeRegressor(max_depth = max_depth, max_leaf_nodes = max_leaf_nodes)
            regr.fit (X_train_bar, y_train)
        
        elif model == "neural_network":
            regr = MLPRegressor (hidden_layer_sizes = hidden_layer_sizes, alpha = alpha_NN, learning_rate = learning_rate_h, max_iter = max_iter_h, random_state = random_state_h, verbose = verbose_h)
            regr.fit (X_train_bar, y_train)
            
        else:
            pass
        
        return regr
    
    def get_rmse (self, regr, X_train_bar, y_train, X_test_bar, y_test):
        
        training_predicted_label = self.get_regression_predictions(regr, X_train_bar)
        training_rmse = self.get_RMSE(training_predicted_label, y_train)
        
        test_predicted_label = self.get_regression_predictions(regr, X_test_bar)   
        test_rmse = self.get_RMSE(test_predicted_label, y_test)
        
        return (training_rmse, test_rmse)
    
    def compute_error_over_Kset (self, file, model, l2_penalty = 0, max_depth = 0, max_leaf_nodes = 0, alpha_NN = 0, hidden_layer_sizes = ()):
        """
            - Purpose: 
                We have k pairs (train_set, test_set).
                  + With each pair, we need to compute RMSE on test_set (and compare with train_set).
                  + With models have hyper-parameters:
                      Iterate over a range of all necessary hyper-param values, after that we need to compute 
                      average error over K sets at each iteration (using only one l2_penalty (in case ridge, lasso)).
                
            - Input: 
                + model: may be one of "linear", "ridge", "lasso"
                + l2_penalty: use only for ridge, lasso
            - Return: a tuple contains <the average of training error> and <the average of test error over>
                K <training and testing> iterations.
        """
        
        """ In each iteration, we need to store training error and test error into 2 separare lists"""
        train_err = []
        test_err  = []
        
        print ("\nResult over K =", self.K, "iteration")
        file.write ("\nResult over K = %d iteration\n" % (self.K))
        if model == "linear":
            print ("train_rmse test_rmse")
            file.write ("train_rmse\ttest_rmse\n")
            
        elif model in ["ridge", "lasso"]:
            print ("log10(penalty) train_rmse test_rmse")
            file.write ("log10(penalty)\ttrain_rmse\ttest_rmse\n")
            
        elif model in ["basic_tree"]:
            print ("max_depth max_leaf_nodes training_rmse test_rmse")
            file.write ("max_depth\tmax_leaf_nodes\ttraining_rmse\ttest_rmse\n")
            
        elif model in ["neural_network"]:
            print ("log10(alpha_NN) no_hidden_layer no_unit_in_a_layer training_rmse test_rmse")
            file.write ("log10(alpha_NN)\tno_hidden_layer\tno_unit_in_a_layer\ttraining_rmse\ttest_rmse\n")                      
            
        else:
            pass
            
        for i in range (self.K):
            
            """ Get data, label"""
            X_train = self.X_train_set[i]
            y_train = self.y_train_set[i]
            
            X_test  = self.X_test_set[i]
            y_test  = self.y_test_set[i]
            
            """ Multi-layer Perceptron is sensitive to feature scaling => need to scale our data"""
            """ Particularly, Standardize features by removing the mean and scaling to unit variance"""
            if model in ["neural_network"]:
                scaler = StandardScaler()  
                scaler.fit(X_train)  
                #print ("Before scaling: ", X_train)
                X_train = scaler.transform(X_train)  
                #print ("After scaling: ", X_train)
                # apply same transformation to test data
                X_test = scaler.transform(X_test)  
            
            """ Get the expanded data for training"""
            X_train_bar = self.get_expand_data (X_train)
            X_test_bar  = self.get_expand_data (X_test)
        
            """ Training model"""
            regr = self.regression (model, X_train_bar, y_train, l2_penalty = l2_penalty, max_depth = max_depth, max_leaf_nodes = max_leaf_nodes, alpha_NN = alpha_NN, hidden_layer_sizes = hidden_layer_sizes)
            
            """ Compute error for a iteration"""
            (training_rmse, test_rmse) = self.get_rmse (regr, X_train_bar, y_train, X_test_bar, y_test)
            
            if model in ["linear"]:
                print (training_rmse, test_rmse)
                file.write ("%f\t%f\n" % (training_rmse, test_rmse))
                
            elif model in ["ridge", "lasso"]:
                print (l2_penalty, training_rmse, test_rmse)
                file.write ("%d\t%f\t%f\n" % (np.log10(l2_penalty), training_rmse, test_rmse))
            
            elif model in ["basic_tree"]:
                print (max_depth, max_leaf_nodes, training_rmse, test_rmse)
                file.write ("%d\t%d\t%f\t%f\n" % (max_depth, max_leaf_nodes, training_rmse, test_rmse))
                
            elif model in ["neural_network"]:
                # TODO: Need to care about alpha
                no_hidden_layer_h = hidden_layer_sizes[0]
                no_unit_in_a_layer_h = len (hidden_layer_sizes)
                print (np.log10(alpha_NN), no_hidden_layer_h, no_unit_in_a_layer_h, training_rmse, test_rmse)
                file.write ("%d\t%d\t%d\t%f\t%f\n" % (np.log10(alpha_NN), no_hidden_layer_h, no_unit_in_a_layer_h, training_rmse, test_rmse))
            
            else:
                pass
            
            
            """ Append errors to the lists"""
            train_err.append (training_rmse)
            test_err.append (test_rmse)
            
        """ Compute average error over K sets"""
        return (np.mean (train_err), np.mean (test_err))
    
    def visualize_tree(self, tree, regr, fn="dt"):
        """
            - Purpose: Create tree based on .dot file using graphviz.
                + NOTE: This function only create .dot file, if we want to create graph => using "plot_tree" in "Plot_lib.py"
                        or "http://webgraphviz.com/"
            - Args:
                + tree -- scikit-learn Decision Tree.
                + fn -- [string], root of filename, default `dt`.
        """
        
        dotfile = fn + ".dot"
        
        try:
            with open(dotfile, 'w') as f:
                
                feature_names_arr = ["1"] + (self.features)#["1", "manufacture_code","rep_model_code","car_code","model_code","vehicle_mile","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]
                
                #target_name = ["price"]
                tree.export_graphviz (regr, out_file = f, max_depth = max_depth_graphviz_h, feature_names=feature_names_arr, filled=True, rounded=True, special_characters=True)  
        except:
            raise IOError ("Error in opening file!\nNote that use are using %s machine." %(machine))
         
     
    def choose_hyper_param (self, file):
        """
            - Purpose: Based on the average test error over cross-validation, we have to select which hyper parameter
                will be applied for our model. We will choose which one has the smallest average test error.
                + The regression models in "self.list_regression_model" will be tested 
            - Input:
            - Return:
        """
        print ("\nCHOOSE HYPER-PARAMETER:\n")
        file.write ("\nCHOOSE HYPER-PARAMETER:\n")
        
        """ Store {model, [optimal hyper-param]} into a dictionary"""
        # TODO: return the dictionary, to use as arguments of "print_final_model" function
        hyper_param_dict = {}
        
        """ Store mean_rmse into the below file for drawing purpose"""
        mean_rmse_error_file = open (mean_rmse_error_file_name, "w")
        
        """ Select hyper-parameter for each regression model"""
        for model in self.list_regression_model:
            
            """ Initialize the "final" of RMSE"""
            if choose_hyperparam == 0:
                final_rmse = min_rmse_h #10**6
            
            selected_i = 0
            selected_j = 0
            
            
            print ("\n\n --- Model: ", model, "---")
            file.write ("\n\n --- Model: %s ---\n" % (model))
            
            if model in ["linear"]:
                (mean_train_err, mean_test_err) = self.compute_error_over_Kset (file = file, model = model)
                print ("\nmean_train_err  mean_test_err")
                file.write ("\nmean_train_err\tmean_test_err\n")
                print (mean_train_err, mean_test_err)
                file.write ("%f\t%f\n" % (mean_train_err, mean_test_err))
                
                mean_rmse_error_file.write ("%f\t%f\n" % (mean_train_err, mean_test_err))
                
            elif model in ["ridge", "lasso"]: 
                """ Initialize an array of penalties will be used to select the suitable hyper-parameter"""
                l2_penalty_array = l2_penalty_array_h.copy() #np.logspace (-10, 4, num = no_penalties)
                len_penalty_array = len (l2_penalty_array)  
                
                
                """ Loop over all penalties"""
                for i in range (len_penalty_array):
                    (mean_train_err, mean_test_err) = self.compute_error_over_Kset (file = file, model = model, l2_penalty = l2_penalty_array[i])
                    
                    print ("\nmean_train_err  mean_test_err")
                    file.write ("\nmean_train_err\tmean_test_err\n")
                    
                    """ Choose the final test error"""
                    
                    if choose_hyperparam == 0 and  mean_test_err < final_rmse: 
                        """ Choose the model with minimum rmse"""
                        (final_rmse, selected_i) = (mean_test_err, i)
                        
                    elif choose_hyperparam == 1: 
                        """ Choose the model with significant improvement of rmse"""
                        if i == 0:
                            initial_rmse = mean_test_err
                            final_rmse = mean_test_err
                        elif self.is_significant_improved (initial_rmse, mean_test_err, significant_value_h) == 1:
                            if mean_test_err < final_rmse:
                                (final_rmse, selected_i) = (mean_test_err, i)
                            # Recently, just choose the model with minimum rmse and has a significant improvement from initial model.                            
                            # TODO: what happen if the next model performance get still be significant with initial value but not enough 
                            # sigificant with previous model???
                    else:
                        pass
                      
                    
                    print (mean_train_err, mean_test_err)
                    file.write ("%f\t%f\n" % (mean_train_err, mean_test_err))
                    mean_rmse_error_file.write ("%f\t%f\n" % (mean_train_err, mean_test_err))
                
                print ("\n\nFinal mean_test_rmse of", model, "over all hyper-param: ", final_rmse, "with penalty:", l2_penalty_array[selected_i])
                file.write ("\n\nFinal mean_test_rmse of %s over all hyper-param is %f with with hyper-param: %f\n" % (model, final_rmse, l2_penalty_array[selected_i]))
                
                """ Store hyper-param into a dictionary"""
                hyper_param_dict[model] = l2_penalty_array[selected_i]
                
            elif model in ["basic_tree"]:
                """ Initiate 2 list of hyper-parameters"""
                max_depth_list      = max_depth_list_h #[5, 10, 20, 50, 100, 200, 10**5]
                max_leaf_nodes_list = max_leaf_nodes_list_h #[5, 10, 20, 50, 100, 200, 10**5]
                
                len_max_depth_list = len (max_depth_list)
                len_max_leaf_nodes_list = len (max_leaf_nodes_list)
                
                """ Loop over all values of max_depth and max_leaf_nodes of a decision tree"""
                break_flag = 0
                for i in range (len_max_depth_list):
                    # Below TODO:
                    if break_flag == 1:
                        break_flag = 0
                        break
                    for j in range (len_max_leaf_nodes_list):
                        (mean_train_err, mean_test_err) = self.compute_error_over_Kset (file = file, model = model, max_depth = max_depth_list[i], max_leaf_nodes = max_leaf_nodes_list[j])
                        
                        print ("\nmean_train_err  mean_test_err")
                        file.write ("\nmean_train_err\tmean_test_err\n")
                           
                        print (mean_train_err, mean_test_err)
                        file.write ("%f\t%f\n" % (mean_train_err, mean_test_err))
                        mean_rmse_error_file.write ("%f\t%f\n" % (mean_train_err, mean_test_err))
                        
                        """ Choose the final test error"""
                        if choose_hyperparam == 0: 
                            """ Choose the model with minimum rmse"""
                            if mean_test_err < final_rmse:
                                (final_rmse, selected_i, selected_j) = (mean_test_err, i, j)
                                
                        elif choose_hyperparam == 1: 
                            """ Choose the model with significant improvement of rmse"""
                            if i == 0 and j == 0:
                                initial_rmse = mean_test_err
                                final_rmse = mean_test_err
                            elif self.is_significant_improved (initial_rmse, mean_test_err, significant_value_h) == 1:# and mean_test_err < final_rmse:
                                (final_rmse, selected_i, selected_j) = (mean_test_err, i, j)
                                break_flag = 1
                                break
                                # Recently, just choose the model with minimum rmse and has a significant improvement from initial model.                            
                                # TODO: what happen if the next model performance get still be significant with initial value but not enough 
                                # sigificant with previous model???
                            #print ("test: %f\t%.2f\n", initial_rmse, 1 - mean_test_err / initial_rmse)
                        else:
                            pass
                        
                print ("\n\nFinal mean_test_rmse of", model, "over all hyper-param: ", final_rmse, "with max_depth:", max_depth_list[selected_i], ", and max_leaf_nodes:", max_leaf_nodes_list[selected_j])
                file.write ("\n\nFinal mean_test_rmse of %s over all hyper-param is %f with max_depth: %d, and max_leaf_nodes: %d\n" % (model, final_rmse, max_depth_list[selected_i], max_leaf_nodes_list[selected_j]))
                
                """ Store hyper-param into a dictionary"""
                hyper_param_dict[model] = (max_depth_list[selected_i], max_leaf_nodes_list[selected_j])
                
            elif model in ["neural_network"]:
                
                
                if l2_regularization_flag == 1:
                    
                    list_alpha = list_alpha_h
                    hidden_layers_sizes = list_hidden_layer_sizes_h[0]
                    len_list_alpha = len (list_alpha)
                    
                    for i in range (len_list_alpha):
                        (mean_train_err, mean_test_err) = self.compute_error_over_Kset (file = file, model = model, alpha_NN = list_alpha[i], hidden_layer_sizes = hidden_layers_sizes)
                        
                        print ("\nmean_train_err  mean_test_err")
                        file.write ("\nmean_train_err\tmean_test_err\n")
                           
                        print (mean_train_err, mean_test_err)
                        file.write ("%f\t%f\n" % (mean_train_err, mean_test_err))
                        mean_rmse_error_file.write ("%f\t%f\n" % (mean_train_err, mean_test_err))
                        
                        """ Choose the final test error"""
                        if choose_hyperparam == 0: 
                            """ Choose the model with minimum rmse"""
                            if mean_test_err < final_rmse:
                                (final_rmse, selected_i) = (mean_test_err, i)
                                
                        elif choose_hyperparam == 1: 
                            """ Choose the model with significant improvement of rmse"""
                            if i == 0:
                                initial_rmse = mean_test_err
                                final_rmse = mean_test_err
                            elif self.is_significant_improved (initial_rmse, mean_test_err, significant_value_h) == 1:
                                if mean_test_err < final_rmse:
                                    (final_rmse, selected_i) = (mean_test_err, i)
                                # Recently, just choose the model with minimum rmse and has a significant improvement from initial model.                            
                                # TODO: what happen if the next model performance get still be significant with initial value but not enough 
                                # sigificant with previous model???
                            #print ("test: %f\t%.2f\n", initial_rmse, 1 - mean_test_err / initial_rmse)
                        else:
                            pass
                    
                    print ("\n\nFinal mean_test_rmse of", model, "over all hyper-param is: ", final_rmse, "with alpha: %d\n" % (list_alpha[selected_i]))
                    """ Interesting thing: "(" + ','.join('{}'.format(*k) for k in enumerate(list_hidden_layer_sizes[selected_i])) + ")")"""
                    file.write ("\n\nFinal mean_test_rmse of %s over all hyper-param is %f with alpha: %d\n" % (model, final_rmse, list_alpha[selected_i]))
                    
                    """ Store hyper-param into a dictionary"""
                    hyper_param_dict[model] = list_alpha[selected_i]
                    
                else:  
                    """ Initiate 2 list of hyper-parameters"""
                    list_hidden_layer_sizes      = list_hidden_layer_sizes_h 
                    
                    len_list_hidden_layer_sizes = len (list_hidden_layer_sizes)              
                
                    """ Loop over all values of no. hidden layers and no. units of a hidden layer"""
                    break_flag = 0
                    for i in range (len_list_hidden_layer_sizes):
                        
                        (mean_train_err, mean_test_err) = self.compute_error_over_Kset (file = file, model = model, alpha_NN = list_alpha_h[0], hidden_layer_sizes = list_hidden_layer_sizes[i])
                        
                        print ("\nmean_train_err  mean_test_err")
                        file.write ("\nmean_train_err\tmean_test_err\n")
                           
                        print (mean_train_err, mean_test_err)
                        file.write ("%f\t%f\n" % (mean_train_err, mean_test_err))
                        mean_rmse_error_file.write ("%f\t%f\n" % (mean_train_err, mean_test_err))
                        
                        """ Choose the final test error"""
                        if choose_hyperparam == 0: 
                            """ Choose the model with minimum rmse"""
                            if mean_test_err < final_rmse:
                                (final_rmse, selected_i) = (mean_test_err, i)
                                
                        elif choose_hyperparam == 1: 
                            """ Choose the model with significant improvement of rmse"""
                            if i == 0:
                                initial_rmse = mean_test_err
                                final_rmse = mean_test_err
                            elif self.is_significant_improved (initial_rmse, mean_test_err, significant_value_h) == 1:
                                if mean_test_err < final_rmse:
                                    (final_rmse, selected_i) = (mean_test_err, i)
                                # Recently, just choose the model with minimum rmse and has a significant improvement from initial model.                            
                                # TODO: what happen if the next model performance get still be significant with initial value but not enough 
                                # sigificant with previous model???
                            #print ("test: %f\t%.2f\n", initial_rmse, 1 - mean_test_err / initial_rmse)
                        else:
                            pass
                            
                    print ("\n\nFinal mean_test_rmse of", model, "over all hyper-param is: ", final_rmse, "with (no_hidden_layer, no_unit_in_a_hidden_layer): (%d, %d)\n:" % (len (list_hidden_layer_sizes[selected_i]), list_hidden_layer_sizes[selected_i][0]))
                    """ Interesting thing: "(" + ','.join('{}'.format(*k) for k in enumerate(list_hidden_layer_sizes[selected_i])) + ")")"""
                    file.write ("\n\nFinal mean_test_rmse of %s over all hyper-param is %f with (no_hidden_layer, no_unit_in_a_hidden_layer): (%d, %d)\n" % (model, final_rmse, len (list_hidden_layer_sizes[selected_i]), list_hidden_layer_sizes[selected_i][0]))
                    
                    """ Store hyper-param into a dictionary"""
                    hyper_param_dict[model] = (list_hidden_layer_sizes[selected_i])
                
                
            else:
                pass
        
        mean_rmse_error_file.close()
        return hyper_param_dict
    
    def get_final_model (self, file):
        
        """ Get data, and label in total datset, and then use it with selected parameters from "choose_hyper_param" to train the full model"""
        X_train = self.X_total_set
        y_train = self.y_total_set
        
        """ Get the expanded data for training"""
        X_train_bar = self.get_expand_data (X_train)
        
        """ Initialize hyper-param = 0"""
        l2_penalty = max_depth = max_leaf_nodes = 0
        
        """ Calculate the optimal hyper-param"""
        hyper_param_dict = self.choose_hyper_param(file)
        print ("hyper_param_dict: ", hyper_param_dict)
    
        for model in self.list_regression_model:
            
            if model in ["ridge", "lasso"]:
                l2_penalty = hyper_param_dict[model]
                
            elif model in ["basic_tree"]:
                (max_depth, max_leaf_nodes) = hyper_param_dict[model]
            
            elif model in ["neural_network"]:
                hidden_layer_sizes = hyper_param_dict[model]
            
            """ Training model"""
            # TODO: Recently, we only rebuild model in total datset with basic_tree.
            # Need to think more about the other cases, due to time consuming
            if model in ["basic_tree"]:
                regr = self.regression (model, X_train_bar, y_train, l2_penalty, max_depth, max_leaf_nodes)        
            
            if model in ["basic_tree"]:
                """ Draw tree in case of using Decision tree in type of .dot file. It need to be converted to .png file (as in Plot_lib file)"""   
                self.visualize_tree(tree, regr, fn = dotfile_name_h + str(max_depth) + "_" + str(max_leaf_nodes)) 
            
            # TODO: what to do with other models?
        
        return (l2_penalty, max_depth, max_leaf_nodes)


    def create_model_keras(self):
        print ("Creating model keras...")
        # create model
        input_dim = 24#X_train_bar.shape[1] 
        hidden_layer_sizes = 1#no_layers
        no_units_in_a_layer = 10
        dropout = 0.7

        # Input layer
        model = Sequential()

        # Hidden layers
        for i in range (hidden_layer_sizes):
            model.add(Dense(no_units_in_a_layer, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
            model.add(Dropout(dropout))

        # Output layer
        model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        print ("Finish creating keras model.")
        return model

    def evaluate_model_keras (self):#, file):#, total_X, total_y, list_dropout, hidden_layer_sizes):
        print ("Evaluating model keras...")
        seed = 7
        np.random.seed(seed)

        list_hidden_layer_sizes      = list_hidden_layer_sizes_h 
                    
        len_list_hidden_layer_sizes = len (list_hidden_layer_sizes)
        
        """ Get data, and label in total datset, and then use it with selected parameters from "choose_hyper_param" to train the full model"""
        X_train = self.X_total_set
        y_train = self.y_total_set
        
        """ Get the expanded data for training"""
        X_train_bar = self.get_expand_data (X_train)

        
        estimator = KerasRegressor (build_fn=self.create_model_keras, nb_epoch=1000, batch_size=5, verbose=0)
        kfold = KFold (n_splits=5, random_state=seed)
        results = cross_val_score (estimator, X_train, y_train, cv=kfold)
        print ("Finish validating keras model.")
        return results.mean()
        
        



