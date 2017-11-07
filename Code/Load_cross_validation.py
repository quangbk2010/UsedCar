#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:07:57 2017

@author: quang
"""

from Evaluate_lib import *
import tensorflow as tf

#with open (textfile_result, "w") as f:
with open ("test.txt", "w") as f:
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    cv = Cross_validation (features, output, list_regression_model, K_fold)

    (l2_penalty, max_depth, max_leaf_nodes) = cv.get_final_model(f)
    #err = cv.evaluate_model_keras ()#f)
    #print (err)    

#X_total_set = total_dataset.get_data_matrix (total_dataset.total_dataset, features)
#y_total_set = total_dataset.get_data_array (total_dataset.total_dataset, output)
