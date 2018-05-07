# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:00:00 2017

@author: quangbk2010

    ##################################################################
    ### CALCULATE PERFORMANCE OF MODEL: MAE, RMSE, REL_ERR, SMAPE
    ##################################################################
    
"""
from a_Header import *


def get_mae (predicted_label, actual_label):
    return mean_absolute_error (predicted_label, actual_label)

def get_rmse (predicted_label, actual_label):
    return np.sqrt (mean_squared_error (predicted_label, actual_label))

def get_relative_err (predicted_label, actual_label):
    try:
        return np.mean (np.abs (predicted_label - actual_label)/actual_label) * 100
    except ValueError:
        print ("Denominator == 0!")
    
def get_smape (predicted_label, actual_label):
    """
        Symetric mean absolute percentage error: is an accuracy measure based on relative error (percentage error) to avoid the case that actual_label is 0
    """
    try:
        return np.mean (np.abs (predicted_label - actual_label) / (np.abs (actual_label) + np.abs (predicted_label)) ) * 100
    except ValueError:
        print ("Denominator == 0!")

def get_err (predicted_label, actual_label): 
    rmse = get_rmse (predicted_label, actual_label)
    mae = get_mae (predicted_label, actual_label)
    rel_err = get_relative_err (predicted_label, actual_label)
    smape = get_smape (predicted_label, actual_label)
    return (rmse, mae, rel_err, smape)


