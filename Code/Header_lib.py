# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:56:43 2017

@author: quang

DECLARE VARIABLE, LIBRARY
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
import time
from sklearn import linear_model, ensemble
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, Imputer, LabelEncoder  
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy import stats
from sklearn.neural_network import MLPRegressor
"""from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor"""
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import TransformerMixin
from sklearn import tree
from os import system
import os
import pickle
import sys
import subprocess
from datetime import datetime
from scipy import stats
from scipy.stats import truncnorm
from collections import Counter


""" ------------------------------------------------ What to change frequently------------------------------------ """

""" Running machine"""
machine = "Server" #"Ubuntu" # or "Mac" or "Windows" or "Server"

""" Features"""
output = "price"#"sale_duration" #"price"
#features = ["manufacture_code","rep_model_code","car_code","model_code","vehicle_mile","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]#,"rep_model_code","car_code","model_code",
features = ["manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]# 33 features
#features = ["price", "manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]# try 33 features + price -> predict sale duration
#features = ["manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type","year","trans_mode","vehicle_mile","cylinder_disp","rental_history","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange"] # 26 features

""" What type of dataset we will use"""
dataset =  "full" # "full", or "partial", or "small"

""" Using cross validation or not"""
using_CV_flag = 0

""" Using onehot or not"""
using_one_hot_flag = 1 # 1-yes, 0-no

""" Encode onehot for car indent"""
using_car_ident_flag = 1 # 1-yes, 0-no

""" Add noise to label of dataset or not to verify results"""
add_noise_flag = 0


""" Remove oulier"""
remove_outliers_flag = 0 # 1-yes, 0-no

""" Regression model will be used"""
list_regression_model = ["neural_network"] #["basic_tree", "gd_boosting", "random_forest", "ada_boost"] #["ridge", "lasso"]#, "basic_tree"]#["basic_tree"] ["neural_network"] #

""" Decision tree visualization"""
max_depth_graphviz_h  = 100
max_depth_list_h      = [10**5]
max_leaf_nodes_list_h = [5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10**4, 10**5]

""" Neural Network parameters"""
""" tuple (the number of neurons in the ith hidden layer), keep the no. units is constant and try to find the no. layers"""
# TODO: the effect of alpha
#alpha_h = 10**(-4)
#no_hidden_layer_h = 2 # it equals to n_layers - 2
#no_unit_in_a_layer_h = 10
#hidden_layer_sizes_h = (no_unit_in_a_layer_h,) * no_hidden_layer_h
learning_rate_h = 'adaptive'
max_iter_h = 1000
random_state_h = 0
verbose_h = False
dropout_h = 1.0

""" Neural Network parameters"""
""" tuple (the number of neurons in the ith hidden layer), keep the no. units is constant and try to find the no. layers"""
# TODO: the effect of alpha
l2_regularization_flag = 1

no_penalties_h = 15
list_alpha_h = np.logspace (-10, 4, num = no_penalties_h) #[10**(-4)]

list_no_hidden_layer_h = [2]#1, 2, 3, 4, 5]#, 6, 7, 8, 9, 10, 15, 20]#, 25, 30, 40, 50] # it equals to n_layers - 2 #[60, 100, 150]
list_no_unit_in_a_layer_h = [6000]#10, 100, 500, 1000]
list_dropout_h = [1]#0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


list_hidden_layer_sizes_h = []
for i in range (len(list_no_hidden_layer_h)):
    for j in range (len(list_no_unit_in_a_layer_h)):
        # hidden_layer_sizes is a tuple
        list_hidden_layer_sizes_h.append ( (list_no_unit_in_a_layer_h[j],) * list_no_hidden_layer_h[i])
#print ("test", list_hidden_layer_sizes_h)

""" Using K-fold Cross Validation"""
K_fold = 5

""" Constraint when get data, label"""
constraint_flag = 0 # 1-yes, 0-no 

""" Name of feature to limit the data, label"""
feature_constraint = "manufacture_code"#"sale_state" #

""" Choose if want the target value is in constraint_values or not"""
is_in_range_constraint = 1 # 1-yes, 0-no

""" Popular Manufacture codes"""
popular_manufacture_codes = [101, 102, 103, 104, 105, 107] #[101]#
sale_state = ["Sold-out"] 
feature_constraint_values = popular_manufacture_codes#sale_state #

""" How to choose hyper parameters"""
choose_hyperparam = 0 # 0-choosing by min (err), 1-choosing by significant improvement
significant_value_h = 0.22



""" ------------------------------------------------ What to not change frequently ------------------------------------ """

full_features = ["manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type","actual_advertising_date","sale_date","year","trans_mode","fuel_type","vehicle_mile","price","sale_state","city","district","dealer_name","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact","option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]

full_features_dict = {"manufacture_code":int,"rep_model_code":int,"car_code":int,"model_code":int,"rating_code":str,"car_type":str,"actual_advertising_date":str,"sale_date":str,"year":int,"trans_mode":str,"fuel_type":str,"vehicle_mile":int,"price":int,"sale_state":str,"city":str,"district":str,"dealer_name":str,"cylinder_disp":int,"tolerance_history":int,"sale_history":int,"rental_history":int,"no_severe_accident":int,"no_severe_water_accident":int,"no_moderate_water_accident":int,"total_no_accident":int,"recovery_fee":int,"hits":int,"no_message_contact":int,"no_call_contact":int,"option_navigation":int,"option_sunLoop":int,"option_smartKey":int,"option_xenonLight":int,"option_heatLineSheet":int,"option_ventilationSheet":int,"option_rearSensor":int,"option_curtainAirbag":int,"no_cover_side_recovery":int,"no_cover_side_exchange":int,"no_corrosive_part":int}

if list_regression_model[0] == "basic_tree":
    sub_directory1 = "Basic tree/"
elif list_regression_model[0] == "gd_boosting":
    sub_directory1 = "Gradient boosting/"
elif list_regression_model[0] == "random_forest":
    sub_directory1 = "Random forest/"
elif list_regression_model[0] == "ada_boost":
    sub_directory1 = "Adaboost/"
elif list_regression_model[0] == "ridge":
    sub_directory1 = "Ridge regression/"
elif list_regression_model[0] == "lasso":
    sub_directory1 = "Lasso regression/"
elif list_regression_model[0] == "linear":
    sub_directory1 = "Linear/"
elif list_regression_model[0] == "neural_network":
    sub_directory1 = "Neural network/"
else:
    raise ValueError ("%s not exist!" %(list_regression_model[0]))

if machine == "Ubuntu":
    directory_h = "/media/sf_Projects/Car Price Estimation Project/Results/Simple case/" + sub_directory1
elif machine in ["Mac", "Windows"]:
    directory_h = "D:/Projects/Car Price Estimation Project/Results/Simple case/" + sub_directory1
elif machine == "Server":
    directory_h = "../Results/" + sub_directory1


if dataset == "full":
    """ Dataset length"""
    input_no = 239444# 239472 #
    dataset_excel_file = '../Data/used_car_Eng_pre_processing1.xlsx'
elif dataset == "partial":
    input_no = 8000
    dataset_excel_file = '../Data/used_car_Eng_pre_processing1_partial dataset.xlsx'
elif dataset == "small":
    input_no = 14
    dataset_excel_file = '../Data/used_car_Eng_pre_processing1_small dataset.xlsx'

print ('data set length:', input_no)
print ("Output:", output)

""" These below parameters used in Validation"""
data_training_percentage = 0.8
data_training_length     = int (0.5 + input_no * data_training_percentage)
#print ('data_training_length', data_training_length)

data_validation_percentage = 0.0
data_validation_length     = int (0.5 + input_no * data_validation_percentage)
#print ('data_validation_length', data_validation_length)

data_test_percentage = 0.2
data_test_length     = int (0.5 + input_no * data_test_percentage)
#print ('data_test_length', data_test_length)


if using_one_hot_flag == 0:
    feature_coding = "Without onehot"
else:
    feature_coding = "With onehot"

if using_car_ident_flag == 0:
    feature_need_encoding = ["manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type", "trans_mode", "fuel_type"]
else:
    feature_need_encoding = ["car_type", "trans_mode", "fuel_type"]

feature_need_label = ["car_type", "trans_mode", "fuel_type"]
feature_need_impute = ["rating_code"]#, "car_type"]
feature_need_remove_outlier = ["vehicle_mile", "no_click", "recovery_fee"]#, "price"]

# list of features whether it needs remove outliers 
feature_need_not_remove_outlier = [feature for feature in features if feature not in feature_need_remove_outlier] 
car_ident = ["manufacture_code","rep_model_code","car_code","model_code","rating_code"]
features_remove_car_ident = [feature for feature in features if feature not in car_ident] 

# list of features whether it needs one-hot encode
feature_need_encoding = ["car_type", "trans_mode", "fuel_type"]
features_not_need_encoding = [feature for feature in features_remove_car_ident if feature not in feature_need_encoding] 

strategy_h = "most_frequent"


""" Calculate RMSE"""
min_rmse_h = 10**6 # set to a large value

""" Ridge and Lasso parameters"""
no_penalties_h = 15
l2_penalty_array_h = np.logspace (-10, 4, num = no_penalties_h)
#really_large_penalty = np.array ([10**10])
#np.concatenate ((l2_penalty_array_h, really_large_penalty), axis = 0)


""" Constraint"""
if constraint_flag == 0:
    sub_directory2 = "/No constraint [All data]"
    constraint_name = "No constraint"
else:
    sub_directory2 = "/Constraint [Categories]"
    constraint_name = "Constraint." + feature_constraint + " = " + str (feature_constraint_values)

""" Result file name"""
pre_file_name = directory_h + dataset + " results/" + feature_coding + sub_directory2 + "/"# + "/[" + dataset + "][" + feature_coding + "][K = " + str(K_fold) + "][" + constraint_name + "]"
if using_CV_flag == 1:
    pre_file_name += "Kfold_CV/"
else:
    pre_file_name += "Validation/"

textfile_result = pre_file_name + "All results.txt" # " Result " + list_regression_model[0] + "_testing.txt"

dotfile_name_h  = pre_file_name + " Best basic tree_"

""" File to write results of mean_rmse over K-fold CV. This file mainly used for drawing figure"""

mean_error_file_name = pre_file_name + "Mean error.txt" # "Mean_rmse over K-fold CV.txt"#" Mean_rmse over K-fold CV_for_testing.txt"

x_embed_file_name = pre_file_name + "Y embeded.txt" # store the vector after applying car2vect
x_ident_file_name = pre_file_name + "x identification.txt" # store the vector after applying car2vect

y_predict_file_name = pre_file_name + "predicted " + output + ".txt" # store predicted value

total_set_file_name = pre_file_name + "total set.txt" # store the data set after shuffle total numpy array
train_set_file_name = pre_file_name + "train set.txt" # store the training set after shuffle total numpy array
test_set_file_name = pre_file_name + "test set.txt" # store the test set after shuffle total numpy array
