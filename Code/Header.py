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
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor

# Scipy libraries
import scipy
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import truncnorm

# Tensorflow libraries
# Determine whether use gpu or not
use_gpu = True 
if use_gpu == False:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import device_lib

# For plotting figures
import matplotlib.pyplot as plt

# Other libraries
import codecs
import tqdm 
from tqdm import tqdm


#print ("{0}, {1}, {2}, {3}".format (np.__version__, pd.__version__, scipy.__version__, tf.__version__))

np.set_printoptions (threshold = np.nan)
pd.set_option('display.max_columns', None)

dataset_type = "new" # "old", "new"

################################
### What to change frequently
################################

""" Running machine"""
machine = "Server" #"Ubuntu" # or "Mac" or "Windows" or "Server"

""" Features"""
if dataset_type == "old":
    #features = ["maker_code","class_code","car_code","model_code","grade_code","year","vehicle_mile","cylinder_disp","rental_history","recovery_fee","views","option_sunLoop"]# Try to remove irrelevant features -> not helpful???
    features = ["maker_code","class_code","car_code","model_code","grade_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","views","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part", "city", "district", "dealer_name","year_diff","adv_month"]# 33 features + dealer area info. = 36 features [+ "year_diff","adv_month" = 38]
    #features = ["price", "maker_code","class_code","car_code","model_code","grade_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","views","no_message_contact","no_call_contact", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]# try 33 features + price -> predict sale duration
else:
    #features = ["class_code","car_code","model_code","car_type","year","vehicle_mile","reg_year","year_diff"] # NOTE Try with KNN for price prediction #"maker_code",
    #features = ["day_diff","views","price","reg_year","class_code","vehicle_mile","car_type","trading_complex","model_code"] # NOTE Try with KNN for sales duration prediction 
    features = ["maker_code","class_code","car_code","model_code","grade_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","views","no_message_contact","no_call_contact","option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part","no_structure_exchange","recovery_fee","branch","affiliate_code","region","trading_complex","min_price","max_price","refund","vain_effort","guarantee","selected_color","mortgage","tax_unpaid","interest","reg_year","reg_month","day_diff","adv_month"] # 51 features, price prediction, REMOVE: seller_id, trading_firm_id, input_color, ADD: adv_month,reg_year # NOTE: best until 2018-03-05, replaced year_diff by day_diff in 2018-03-12

    #features = ["maker_code","class_code","car_code","model_code","grade_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","views","no_message_contact","no_call_contact","option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part","no_structure_exchange","recovery_fee","branch","affiliate_code","region","trading_complex","min_price","max_price","refund","vain_effort","guarantee","selected_color","mortgage","tax_unpaid","interest","reg_year","reg_month", "day_diff", "adv_month","trading_firm_id","seller_id","price"] # 54 features, SD prediction, ADD: price, seller_id, trading_firm_id 
    #features = ["maker_code","class_code","car_code","model_code","grade_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","views","option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part","no_structure_exchange","region","trading_complex","selected_color","tax_unpaid","reg_year","reg_month","day_diff","adv_month","price"] # 33 features, SD prediction
    #features = ["day_diff","reg_year","views","price","vehicle_mile","car_code","trading_complex","model_code","region","grade_code","reg_month","year","selected_color","adv_month"] # 14 features, SD prediction


""" Using cross validation or not"""
using_CV_flag = 0

""" Using onehot or not"""
using_one_hot_flag = 1 # 1-yes, 0-no #####NOTE

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

if dataset_type == "old":
    full_features = ["maker_code","class_code","car_code","model_code","grade_code","car_type","actual_advertising_date","sale_date","year","trans_mode","fuel_type","vehicle_mile","price","sale_state","city","district","dealer_name","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","recovery_fee","views","no_message_contact","no_call_contact","option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part"]

    #full_features_dict = {"maker_code":int,"class_code":int,"car_code":int,"model_code":int,"grade_code":str,"car_type":str,"actual_advertising_date":str,"sale_date":str,"year":int,"trans_mode":str,"fuel_type":str,"vehicle_mile":int,"price":int,"sale_state":str,"city":str,"district":str,"dealer_name":str,"cylinder_disp":int,"tolerance_history":int,"sale_history":int,"rental_history":int,"no_severe_accident":int,"no_severe_water_accident":int,"no_moderate_water_accident":int,"total_no_accident":int,"recovery_fee":int,"views":int,"no_message_contact":int,"no_call_contact":int,"option_navigation":int,"option_sunLoop":int,"option_smartKey":int,"option_xenonLight":int,"option_heatLineSheet":int,"option_ventilationSheet":int,"option_rearSensor":int,"option_curtainAirbag":int,"no_cover_side_recovery":int,"no_cover_side_exchange":int,"no_corrosive_part":int}
else:
    full_features = ["plate_num","maker_code","class_code","car_code","model_code","grade_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","views","no_message_contact","no_call_contact","option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part","no_structure_exchange","recovery_fee","branch","affiliate_code","region","trading_complex","trading_firm_id","seller_id","price","min_price","max_price","first_adv_date","sale_date","refund","vain_effort","guarantee","first_registration","selected_color","input_color","mortgage","tax_unpaid","interest"] #recovery fee (based on 10 accident days)

    #full_features_dict = {"plate_num":str,"maker_code":int,"class_code":int,"car_code":int,"model_code":int,"grade_code":int,"car_type":str,"year":int,"trans_mode":str,"fuel_type":str,"vehicle_mile":int,"cylinder_disp":int,"tolerance_history":int,"sale_history":int,"rental_history":int,"no_severe_accident":int,"no_severe_water_accident":int,"no_moderate_water_accident":int,"total_no_accident":int,"views":int,"no_message_contact":int,"no_call_contact":int,"option_navigation":int,"option_sunLoop":int,"option_smartKey":int,"option_xenonLight":int,"option_heatLineSheet":int,"option_ventilationSheet":int,"option_rearSensor":int,"option_curtainAirbag":int,"no_cover_side_recovery":int,"no_cover_side_exchange":int,"no_corrosive_part":int,"no_structure_exchange":int,"recovery_fee":int,"branch":str,"affiliate_code":int,"region":str,"trading_complex":str,"trading_firm_id":str,"seller_id":str,"price":int,"min_price":int,"max_price":int,"first_adv_date":str,"sale_date":str,"refund":str,"vain_effort":str,"guarantee":str,"first_registration":str,"selected_color":str,"input_color":str,"mortgage":int,"tax_unpaid":int,"interest":int}
    


""" Popular Manufacture codes"""
popular_maker_codes = [101, 102, 103, 104, 105, 107] #[101]#


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

print ("0.", device_lib.list_local_devices())


