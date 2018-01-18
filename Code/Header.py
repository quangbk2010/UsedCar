# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:00:00 2017

@author: quang

    #################################
    ### DECLARE VARIABLES, LIBRARIES
    #################################

"""

# General libraries
from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
import time
from os import system
import os
import sys
import subprocess
from datetime import datetime
from collections import Counter

# For saving model 
import pickle
import tables

# For parsing arguments that passed to python file
import argparse

# Sklearn libraries
from sklearn import linear_model, ensemble, tree
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, Imputer, LabelEncoder  
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.base import TransformerMixin
from sklearn.neural_network import MLPRegressor

# Scipy libraries
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import truncnorm

# Tensorflow libraries
import tensorflow as tf
import tensorflow.contrib.slim as slim

# For plotting figures
import matplotlib.pyplot as plt


################################
### What to change frequently
################################

""" Running machine"""
machine = "Server" #"Ubuntu" # or "Mac" or "Windows" or "Server"

""" Features"""
features = ["manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]# 33 features
#features = ["price", "manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]# try 33 features + price -> predict sale duration
#features = ["manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type","year","trans_mode","vehicle_mile","cylinder_disp","rental_history","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange"] # 26 features


""" Using cross validation or not"""
using_CV_flag = 0

""" Using onehot or not"""
using_one_hot_flag = 1 # 1-yes, 0-no

""" Add noise to label of dataset or not to verify results"""
add_noise_flag = 0

""" Remove oulier"""
remove_outliers_flag = 0 # 1-yes, 0-no

""" Regression model will be used"""
list_regression_model = ["neural_network"] #["basic_tree", "gd_boosting", "random_forest", "ada_boost"] #["ridge", "lasso"]#, "basic_tree"]#["basic_tree"] ["neural_network"] 

""" Decision tree visualization"""
max_depth_graphviz_h  = 100
max_depth_list_h      = [10**5]
max_leaf_nodes_list_h = [5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10**4, 10**5]





##################################
### What to not change frequently
##################################

full_features = ["manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type","actual_advertising_date","sale_date","year","trans_mode","fuel_type","vehicle_mile","price","sale_state","city","district","dealer_name","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","no_click","no_message_contact","no_call_contact","option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]

full_features_dict = {"manufacture_code":int,"rep_model_code":int,"car_code":int,"model_code":int,"rating_code":str,"car_type":str,"actual_advertising_date":str,"sale_date":str,"year":int,"trans_mode":str,"fuel_type":str,"vehicle_mile":int,"price":int,"sale_state":str,"city":str,"district":str,"dealer_name":str,"cylinder_disp":int,"tolerance_history":int,"sale_history":int,"rental_history":int,"no_severe_accident":int,"no_severe_water_accident":int,"no_moderate_water_accident":int,"total_no_accident":int,"recovery_fee":int,"hits":int,"no_message_contact":int,"no_call_contact":int,"option_navigation":int,"option_sunLoop":int,"option_smartKey":int,"option_xenonLight":int,"option_heatLineSheet":int,"option_ventilationSheet":int,"option_rearSensor":int,"option_curtainAirbag":int,"no_cover_side_recovery":int,"no_cover_side_exchange":int,"no_corrosive_part":int}


""" Popular Manufacture codes"""
popular_manufacture_codes = [101, 102, 103, 104, 105, 107] #[101]#


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


if using_one_hot_flag == 0:
    feature_coding = "Without onehot"
else:
    feature_coding = "With onehot"


