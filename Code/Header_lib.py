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
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy import stats
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn import tree
from os import system
import subprocess
from datetime import datetime


""" ------------------------------------------------ What to change frequently------------------------------------ """

""" Running machine"""
machine = "Server" #"Ubuntu" # or "Mac" or "Windows" or "Server"

""" Features"""
output = "sale_duration" #"price"
features = ["manufacture_code","rep_model_code","car_code","model_code","vehicle_mile","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]#,"rep_model_code","car_code","model_code",
#features = ["manufacture_code","rep_model_code","vehicle_mile","no_severe_accident","total_no_accident","no_click","option_sunLoop","option_smartKey","option_xenonLight","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_exchange"]#,,"rep_model_code","car_code","model_code","option_heatLineSheet","no_corrosive_part"

""" What type of dataset we will use"""
dataset = "Full dataset" # "Full dataset" #or "Partial dataset" # or "Partial dataset_testing"


""" Using onehot or not"""
using_one_hot_flag = 0 # 1-yes, 0-no

""" Regression model will be used"""
list_regression_model = ["neural_network"] #["ridge", "lasso"]#, "basic_tree"]#["basic_tree"] 

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
dropout_h = 1

""" Neural Network parameters"""
""" tuple (the number of neurons in the ith hidden layer), keep the no. units is constant and try to find the no. layers"""
# TODO: the effect of alpha
l2_regularization_flag = 1

no_penalties_h = 15
list_alpha_h = np.logspace (-10, 4, num = no_penalties_h) #[10**(-4)]

list_no_hidden_layer_h = [9] # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50] # it equals to n_layers - 2 #[60, 100, 150]
list_no_unit_in_a_layer_h = [10] #,100]


list_hidden_layer_sizes_h = []
for i in range (len(list_no_hidden_layer_h)):
    for j in range (len(list_no_unit_in_a_layer_h)):
        # hidden_layer_sizes is a tuple
        list_hidden_layer_sizes_h.append ( (list_no_unit_in_a_layer_h[j],) * list_no_hidden_layer_h[i])
#print ("test", list_hidden_layer_sizes_h)

""" Using K-fold Cross Validation"""
K_fold = 5

""" Constraint when get data, label"""
constraint_flag = 1 # 1-yes, 0-no

""" Name of feature to limit the data, label"""
feature_constraint = "sale_state" #"manufacture_code"

""" Choose if want the target value is in constraint_values or not"""
is_in_range_constraint = 1 # 1-yes, 0-no

""" Popular Manufacture codes"""
popular_manufacture_codes = [101, 102, 103, 104, 105] #[101]#
sale_state = ["Sold-out"] 
feature_constraint_values = sale_state #popular_manufacture_codes

""" How to choose hyper parameters"""
choose_hyperparam = 0 # 0-choosing by min (err), 1-choosing by significant improvement
significant_value_h = 0.22



""" ------------------------------------------------ What to not change frequently ------------------------------------ """

full_features = ["manufacture_name","representative_name","car_name","model_name","rating_name","manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type","init_advertising_reg_date","actual_advertising_date","sale_date","year","trans_mode","fuel_type","vehicle_mile","price","sale_state","city","district","dealer_name","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","no_click","online_registration_date","no_message_contact","no_call_contact","option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part","gbn_code1","accident_day1","insurance_amt1","component_amt1","wage_amt1","painting_amt1","gbn_code2","accident_day2","insurance_amt2","component_amt2","wage_amt2","painting_amt2","gbn_code3","accident_day3","insurance_amt3","component_amt3","wage_amt3","painting_amt3","gbn_code4","accident_day4","insurance_amt4","component_amt4","wage_amt4","painting_amt4","gbn_code5","accident_day5","insurance_amt5","component_amt5","wage_amt5","painting_amt5","gbn_code6","accident_day6","insurance_amt6","component_amt6","wage_amt6","painting_amt6","gbn_code7","accident_day7","insurance_amt7","component_amt7","wage_amt7","painting_amt7","gbn_code8","accident_day8","insurance_amt8","component_amt8","wage_amt8","painting_amt8","gbn_code9","accident_day9","insurance_amt9","component_amt9","wage_amt9","painting_amt9","gbn_code10","accident_day10","insurance_amt10","component_amt10","wage_amt10","painting_amt10"]
full_features_dict = {"manufacture_name":str,"representative_name":str,"car_name":str,"model_name":str,"rating_name":str,"manufacture_code":int,"rep_model_code":int,"car_code":int,"model_code":int,"rating_code":str,"car_type":str,"init_advertising_reg_date":str,"actual_advertising_date":str,"sale_date":str,"year":int,"trans_mode":str,"fuel_type":str,"vehicle_mile":int,"price":int,"sale_state":str,"city":str,"district":str,"dealer_name":str,"cylinder_disp":int,"tolerance_history":int,"sale_history":int,"rental_history":int,"no_severe_accident":int,"no_severe_water_accident":int,"no_moderate_water_accident":int,"total_no_accident":int,"recovery_fee":int,"hits":int,"online_registration_date":str,"no_message_contact":int,"no_call_contact":int,"option_navigation":int,"option_sunLoop":int,"option_smartKey":int,"option_xenonLight":int,"option_heatLineSheet":int,"option_ventilationSheet":int,"option_rearSensor":int,"option_curtainAirbag":int,"no_cover_side_recovery":int,"no_cover_side_exchange":int,"no_corrosive_part":int,"gbn_code1":int,"accident_day1":int,"insurance_amt1":int,"component_amt1":int,"wage_amt1":int,"painting_amt1":int,"gbn_code2":int,"accident_day2":int,"insurance_amt2":int,"component_amt2":int,"wage_amt2":int,"painting_amt2":int,"gbn_code3":int,"accident_day3":int,"insurance_amt3":int,"component_amt3":int,"wage_amt3":int,"painting_amt3":int,"gbn_code4":int,"accident_day4":int,"insurance_amt4":int,"component_amt4":int,"wage_amt4":int,"painting_amt4":int,"gbn_code5":int,"accident_day5":int,"insurance_amt5":int,"component_amt5":int,"wage_amt5":int,"painting_amt5":int,"gbn_code6":int,"accident_day6":int,"insurance_amt6":int,"component_amt6":int,"wage_amt6":int,"painting_amt6":int,"gbn_code7":int,"accident_day7":int,"insurance_amt7":int,"component_amt7":int,"wage_amt7":int,"painting_amt7":int,"gbn_code8":int,"accident_day8":int,"insurance_amt8":int,"component_amt8":int,"wage_amt8":int,"painting_amt8":int,"gbn_code9":int,"accident_day9":int,"insurance_amt9":int,"component_amt9":int,"wage_amt9":int,"painting_amt9":int,"gbn_code10":int,"accident_day10":int,"insurance_amt10":int,"component_amt10":int,"wage_amt10":int,"painting_amt10":int}

if list_regression_model[0] == "basic_tree":
    sub_directory1 = "Tree plot/"
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
    directory_h = "Results/" + sub_directory1

if dataset == "Full dataset":
    dataset_excel_file = '../Data/used_car_Eng_2.xlsx'
    """ Dataset length"""
    input_no = 239472
elif dataset == "Partial dataset":
    dataset_excel_file = '../Data/used_car_Eng_small dataset.xlsx'
    input_no = 25458
elif dataset == "Partial dataset_testing":
    dataset_excel_file = '../Data/used_car_Eng_small dataset_for testing.xlsx'
    input_no = 14

print ('data set length:', input_no)
print ("Output:", output)

""" These below parameters used in Validation"""
data_training_percentage = 0.8
data_training_length     = int (0.5 + input_no * data_training_percentage)
#print ('data_training_length', data_training_length)

data_validation_percentage = 0.1
data_validation_length     = int (0.5 + input_no * data_validation_percentage)
#print ('data_validation_length', data_validation_length)

data_test_percentage = 0.1
data_test_length     = int (0.5 + input_no * data_test_percentage)
#print ('data_test_length', data_test_length)


if using_one_hot_flag == 0:
    feature_coding = "Without onehot"
else:
    feature_coding = "With onehot"


feature_need_encoding = ["manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type"]



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
pre_file_name = directory_h + dataset + " results/" + feature_coding + sub_directory2 + "/[" + dataset + "][" + feature_coding + "][K = " + str(K_fold) + "][" + constraint_name + "]"

textfile_result = pre_file_name + "All results.txt" # " Result " + list_regression_model[0] + "_testing.txt"

dotfile_name_h  = pre_file_name + " Best basic tree_"

""" File to write results of mean_rmse over K-fold CV. This file mainly used for drawing figure"""

mean_rmse_error_file_name = pre_file_name + "Mean RMSE.txt" # "Mean_rmse over K-fold CV.txt"#" Mean_rmse over K-fold CV_for_testing.txt"


