# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:52:31 2017

@author: quang
"""

from Used_car_regression import *


training = Training()  

training_data_matrix = training.get_data_matrix (training.get_training_dataset(), features)
training_data_matrix_bar = training.get_expand_data (training_data_matrix)
training_label_array = training.get_data_array (training.get_training_dataset(), output)  

validation_data_matrix = training.get_data_matrix (training.get_validation_dataset(), features)
validation_data_matrix_bar = training.get_expand_data (validation_data_matrix)
validation_label_array = training.get_data_array (training.get_validation_dataset(), output)  

test_data_matrix = training.get_data_matrix (training.get_test_dataset(), features)
test_data_matrix_bar = training.get_expand_data (test_data_matrix)
test_label_array = training.get_data_array (training.get_test_dataset(), output)    

