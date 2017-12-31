# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:57:02 2017

@author: user
"""
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np
import matplotlib.pyplot as plt
from Header_lib import *
import pickle
import os
import sys
from Used_car_regression import *

print ("Advanced tree.")
grid_file = "./Model/[" + dataset  +  "]grid_search_DT.sav_" #'./Model/GradientBoostingTree/[GradientBoostingRegressor] finalized_model.sav'

#############################################################################################
total_dataset = Dataset()
dataset = total_dataset.get_total_dataset()
if output == "price":
    X_total_set = total_dataset.get_data_matrix (total_dataset.get_total_dataset(), features)  
    y_total_set = total_dataset.get_data_array (total_dataset.get_total_dataset(), output) 
    
elif output == "sale_duration":
    #X_total_set = total_dataset.get_data_matrix_with_constraint (total_dataset.get_total_dataset(), features, "sale_state", "Sold-out")  
    X_total_set = dataset[features]
    y_total_set = total_dataset.get_sale_duration_array (dataset) 
    
"""scaler = StandardScaler()  
scaler.fit(X_total_set)  
X_total_set = scaler.transform(X_total_set)"""
print (X_total_set.shape, y_total_set.shape)
len_total_set = X_total_set.shape[0]
training_length = int (0.5 + len_total_set * data_training_percentage)
total_set = np.concatenate ((X_total_set, y_total_set), axis = 1)
  
#################
total_data = total_set [:, 0:total_set.shape[1]-1]
total_label = total_set [:, total_set.shape[1]-1:]

train_data  = total_data[:training_length, :]
train_label = total_label[:training_length, :]


test_data  = total_data[training_length:, :]
test_label = total_label[training_length:, :]

print (test_label[0])
print (train_data.shape, train_label.shape)
print (test_data.shape, test_label.shape)

err_type = "relative_err" #"relative_err","smape" 

#############################################################################################

def get_predicted_label (tree, data):
    return  np.array (tree.predict(data)).reshape (data.shape[0], 1)

def get_relative_err (predicted_label, actual_label):
    """if output == "sale_duration":
        rel_err = np.zeros ((actual_label.shape[0], 1))
        tag = actual_label <=5
        diff_date = dataset["diff_date"]
        #print (tag)
        #sys.exit (-1)
        rel_err [tag] = np.abs (predicted_label - actual_label)/actual_label
        #rel_err = [np.abs (predicted_label[i] - actual_label[i])/actual_label[i]]"""
        
    return np.mean (np.abs (predicted_label - actual_label)/actual_label) * 100

def get_smape (predicted_label, actual_label):
    """
        Symetric mean absolute percentage error: is an accuracy measure based on relative error (percentage error) to avoid the case that actual_label is 0
    """
    if (np.abs (actual_label) + np.abs (predicted_label)).all() == 0:
        return -1
    return np.mean (np.abs (predicted_label - actual_label) / (np.abs (actual_label) + np.abs (predicted_label)) ) * 100

def get_err (predicted_label, actual_label): #err_type
    if err_type == "relative_err":
        return get_relative_err (predicted_label, actual_label)
    elif err_type == "smape":
        return get_smape (predicted_label, actual_label)


def plot_feature_importance():
    # Plot feature importance
    feature_importance = reg_tree.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.axis ([0, 100, 0, 0.1])
    fft_axes = plt.subplot()#1, 2, 2)
    
        
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    feature_names = []
    for i in sorted_idx:
        feature_names.append (features[i])
    plt.yticks(pos, feature_names, fontsize=7)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.tight_layout()
    plt.show()

#############################################################################################
#plot_feature_importance()
####### 
# Basic tree
# Training
def tree_regression (reg_type):
    global grid_file
    grid_file += reg_type
    stime = time.time()
    print ("training...")

    if reg_type == "DecisionTreeRegressor":
        reg_tree = tree.DecisionTreeRegressor() 
        param_grid = [ {"criterion":"mae", "min_samples_split":[5, 10, 20], "max_depth":[10, 20, 30]}]
    elif reg_type == "GradientBoostingRegressor":
        reg_tree = ensemble.GradientBoostingRegressor()
        param_grid = [ {"n_estimators": [10, 100, 500, 1000],"learning_rate": [0.001, 0.01, 0.1, 0.2], "max_depth":[10,20,30]}]
    elif reg_type == "AdaBoostRegressor":
        reg_tree = ensemble.AdaBoostRegressor()
        param_grid = [ {"n_estimators": [10, 100, 500, 1000], "min_samples_split": [5, 10, 20], "max_depth": [10, 20, 30]}]
    elif reg_type == "RandomForestRegressor":
        reg_tree = ensemble.RandomForestRegressor()
        param_grid = [ {"n_estimators": [10, 100, 500, 1000], "min_samples_split": [5, 10, 20], "max_depth": [10, 20, 30]},
                       {"bootstrap": [False], "n_estimators": [10, 100, 500, 1000], "min_samples_split": [5, 10, 20], "max_depth": [10, 20, 30]} 
        ]

    err = make_scorer (get_err, greater_is_better=False)

    if os.path.isfile (grid_file) == False:
        grid_search = GridSearchCV (reg_tree, param_grid, cv = 2, scoring = err)
        grid_search.fit (train_data, train_label)
        pickle.dump (grid_search, open (grid_file, 'wb'))
    else:
        # load the model from disk
        print ("load the gridsearch from disk")
        grid_search = pickle.load(open(grid_file, 'rb'))

    print (grid_search.best_params_)
    print (grid_search.best_estimator_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip (cvres["mean_test_score"], cvres["params"]):
        print (mean_score, params)
    # Testing
    print ("testing...")
    predicted_train_label = get_predicted_label (grid_search, train_data) 
    train_err = get_err (predicted_train_label, train_label)
    
    predicted_test_label = get_predicted_label (grid_search, test_data) 
    test_err = get_err (predicted_test_label, test_label) 
    print ("[" + reg_type  + "] Relative err: train_err: %.3f %%, test_err: %.3f %%" % (train_err, test_err))


for reg_type in ["DecisionTreeRegressor", "GradientBoostingRegressor", "RandomForestRegressor", "AdaBoostRegressor"]:
    print ("test:", reg_type)
    tree_regression(reg_type)

'''

    if reg_type == "DecisionTreeRegressor":
        stime = time.time()
        print ("training...")

        """print ("here")
        for i in range (10):
            sys.stdout.write ("\r" + "test: %d" % (i))
        print ("there")"""

        reg_tree = tree.DecisionTreeRegressor() #criterion="mae",min_samples_split=5,max_depth=20) #max_leaf_nodes=10000)
        param_grid = [ {"min_samples_split":[5, 10, 20], "max_depth":[10, 20, 30]}]
        err = make_scorer (get_err, greater_is_better=False)

        if os.path.isfile (grid_file) == False:
            grid_search = GridSearchCV (reg_tree, param_grid, cv = 5, scoring = err)
            grid_search.fit (train_data, train_label)
            pickle.dump (grid_search, open (grid_file, 'wb'))
        else:
            # load the model from disk
            print ("load the gridsearch from disk")
            grid_search = pickle.load(open(grid_file, 'rb'))

        print (grid_search.best_params_)
        print (grid_search.best_estimator_)
        cvres = grid_search.cv_results_
        for mean_score, params in zip (cvres["mean_test_score"], cvres["params"]):
            print (mean_score, params)
        # Testing
        print ("testing...")
        predicted_train_label = get_predicted_label (grid_search, train_data) 
        train_err = get_err (predicted_train_label, train_label)
        
        predicted_test_label = get_predicted_label (grid_search, test_data) 
        test_err = get_err (predicted_test_label, test_label) 
        print ("[DecisionTreeRegressor] Relative err: train_err: %.3f %%, test_err: %.3f %%" % (train_err, test_err))
                
    
    #######
    # GradientBoostingRegressor
    # Training
    elif reg_type == "GradientBoostingRegressor":
        if os.path.isfile (filename) == False:
            params = {'n_estimators': 1000,'learning_rate': 0.1, 'max_depth':20, 'loss': 'lad'}
            reg_tree = ensemble.GradientBoostingRegressor(**params)
            stime = time.time()
            print ("training...")
            reg_tree.fit (train_data, train_label)
            print("Time for GradientBoostingRegressor learning_rate 0.1 tree fitting: %.3f" % (time.time() - stime))
            
            # save the model to disk
            print ("save the model to disk")
            pickle.dump(reg_tree, open(filename, 'wb'))
        else:
            # load the model from disk
            print ("load the model from disk")
            reg_tree = pickle.load(open(filename, 'rb'))
        
        # Testing
        print ("testing...")
        predicted_train_label = get_predicted_label (reg_tree, train_data) # np.array (reg_tree.predict(test_data)).reshape (test_data.shape[0], 1)
        train_err = get_err (err_type, predicted_train_label, train_label)
        
        predicted_test_label = get_predicted_label (reg_tree, test_data) # np.array (reg_tree.predict(test_data)).reshape (test_data.shape[0], 1)
        test_err = get_err (err_type, predicted_test_label, test_label) #np.mean (np.abs (predicted_label - test_label)/test_label) * 100
        print ("[GradientBoostingRegressor] Relative err: train_err: %.3f %%, test_err: %.3f %%" % (train_err, test_err))
        
        a = np.concatenate ((test_label, predicted_test_label), axis = 1)
        print ("test_label:", a[:7])
        np.savetxt('[GradientBoostingRegressor] predicted_price.txt', a, fmt="%.2f %.2f")
    
    
    #######
    # AdaBoostRegressor
    # Training
    
    elif reg_type == "AdaBoostRegressor":
        reg_tree = ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(criterion="mae",min_samples_split=5,max_depth=20),n_estimators=500)
        stime = time.time()
        print ("training...")
        reg_tree.fit(train_data,train_label)
        print("Time for AdaBoostRegressor tree fitting: %.3f" % (time.time() - stime))
        
        # Testing
        print ("testing...")
        predicted_train_label = get_predicted_label (reg_tree, train_data) # np.array (reg_tree.predict(test_data)).reshape (test_data.shape[0], 1)
        train_err = get_err (err_type, predicted_train_label, train_label)
        
        predicted_test_label = get_predicted_label (reg_tree, test_data) # np.array (reg_tree.predict(test_data)).reshape (test_data.shape[0], 1)
        test_err = get_err (err_type, predicted_test_label, test_label) #np.mean (np.abs (predicted_label - test_label)/test_label) * 100
        print ("[AdaBoostRegressor] Relative err: train_err: %.3f %%, test_err: %.3f %%" % (train_err, test_err))
        
    #######
    # RandomForestRegressor
    # Training    
    
    elif reg_type == "RandomForestRegressor":
        reg_tree = ensemble.RandomForestRegressor(max_leaf_nodes=10000, random_state=2)#max_depth=15  max_leaf_nodes=10000
        stime = time.time()
        print ("training...")
        reg_tree.fit(train_data,train_label)
        print("Time for RandomForestRegressor tree fitting: %.3f" % (time.time() - stime))
        
        # Testing
        print ("testing...")
        predicted_train_label = get_predicted_label (reg_tree, train_data) # np.array (reg_tree.predict(test_data)).reshape (test_data.shape[0], 1)
        train_err = get_err (err_type, predicted_train_label, train_label)
        
        predicted_test_label = get_predicted_label (reg_tree, test_data) # np.array (reg_tree.predict(test_data)).reshape (test_data.shape[0], 1)
        test_err = get_err (err_type, predicted_test_label, test_label) #np.mean (np.abs (predicted_label - test_label)/test_label) * 100
        print ("[RandomForestRegressor] Relative err: train_err: %.3f %%, test_err: %.3f %%" % (train_err, test_err))
'''
