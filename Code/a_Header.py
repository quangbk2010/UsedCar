# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:00:00 2017

@author: quangbk2010

    #################################
    ### DECLARE VARIABLES, LIBRARIES
    #################################

"""

# General libraries
from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
import time
import os
import sys
from datetime import datetime

# For saving model 
import pickle

# For parsing arguments that passed to python file
import argparse

# Sklearn libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, Imputer, LabelEncoder  
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.base import TransformerMixin

# Scipy libraries
import scipy

# Tensorflow libraries
# Determine whether use gpu or not
use_gpu = True 
if use_gpu == True:
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 0-only use GPU0, 1-only use GPU1 (but when check in device_lib the name of existing device, it still be GPU:0, check nvidia-smi -> to see the difference)
    pass #Use all GPUs (also need to specify in Main file to use all)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import device_lib

# Other libraries
from tqdm import tqdm


#print ("{0}, {1}, {2}, {3}".format (np.__version__, pd.__version__, scipy.__version__, tf.__version__))

np.set_printoptions (threshold = np.nan)
pd.set_option('display.max_columns', None)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #0: default-all logs shown, 1-INFO logs, 2-WARNING logs, 3-ERROR logs

dataset_type = "new" 

################################
### What to change frequently
################################

""" Running machine"""
machine = "Server" 

""" Features"""
features = ["maker_code","class_code","car_code","model_code","grade_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","views","no_message_contact","no_call_contact","option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part","no_structure_exchange","recovery_fee","branch","affiliate_code","region","trading_complex","min_price","max_price","refund","vain_effort","guarantee","selected_color","mortgage","tax_unpaid","interest","reg_year","reg_month","day_diff","adv_month"] # 51 features, price prediction, REMOVE: seller_id, trading_firm_id, input_color, ADD: adv_month,reg_year # NOTE: best until 2018-03-05, replaced year_diff by day_diff in 2018-03-12


""" Header name of input file"""
full_features = ["plate_num","maker_code","class_code","car_code","model_code","grade_code","car_type","year","trans_mode","fuel_type","vehicle_mile","cylinder_disp","tolerance_history","sale_history","rental_history","no_severe_accident","no_severe_water_accident","no_moderate_water_accident","total_no_accident","views","no_message_contact","no_call_contact","option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part","no_structure_exchange","recovery_fee","branch","affiliate_code","region","trading_complex","trading_firm_id","seller_id","price","min_price","max_price","first_adv_date","sale_date","refund","vain_effort","guarantee","first_registration","selected_color","input_color","mortgage","tax_unpaid","interest"] #recovery fee (based on 10 accident days)




