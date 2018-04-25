from __future__ import division
import numpy as np
import pandas as pd

from datetime import datetime
import sys
np.set_printoptions (threshold = np.nan)
pd.set_option('display.max_columns', None)
import math
import scipy.stats as stats
from tqdm import tqdm

def compute_cwc (data, gama1, gama2, eta=10, muy=0.95, alpha1=0, alpha2=0):
    len_data = len (data)
    R = np.max (data["price"]) - np.min (data["price"])
    data["lower"]  = data["pred_price"] * (1 - gama1) - alpha1 * data["std"]
    data["higher"] = data["pred_price"] * (1 + gama2) + alpha2 * data["std"]
    mpiw = np.mean (data["pred_price"] * (gama1 + gama2) + (alpha1 + alpha2)*data["std"])
    no_in_range    = np.sum ((data["price"] > data["lower"]) & (data["price"] < data["higher"]))
    
    picp = no_in_range / len_data 
    nmpiw = mpiw / R
    #cwc = nmpiw * (1 + gama (picp, muy) * math.exp (-eta * (picp - muy)))
    cwc = nmpiw * math.exp (-eta * (picp - muy))
    #print ("===%.2f, %.2f, %.2f, %.2f, %.2f, %.4f" % ( gama1, gama2, nmpiw, picp, muy, cwc)) # nmpiw is small as possible, percent is large as posible, cwc is small as possible
    
    return (gama1, gama2, eta, alpha1, alpha2, nmpiw, picp, muy, cwc)

def read_data ():
    data = pd.read_csv ("../Results/price.txt", sep="\t")
    len_data = len (data)
    len1 = int (len_data / 2)
    #print (data.shape)
    data["delta"] = data["pred_price"] - data["price"]    
    # Divide it into 2 part: validation set and test set
    data1 = data [:len1]
    data2 = data [len1:]
    return data1, data2

def determine_pred_interval (data1, eta=10, percent_threshold=0.95):
    muy = percent_threshold # eta: 10, 100, 300
    res = np.empty ((0, 9))
    # Determine the range that ground truth included in the range from the validation set   
    """for gama1 in np.arange (0.05, 0.26, 0.01):#np.arange (0.05, 0.3, 0.05):
        for gama2 in np.arange (0.05, 0.07, 0.01):#np.arange (0.05, 0.3, 0.05):
            for alpha1 in np.arange (0, 1.1, 0.5):
                for alpha2 in np.arange (0, 1.1, 0.5):"""
    a = np.arange (0, 0.26, 0.01)
    b = np.arange (4, 5.1, 0.2)

    for gama1 in  tqdm (a):
        for gama2 in tqdm (a):
            for alpha1 in tqdm (b):
                for alpha2 in tqdm (b):
                    res = np.vstack ([res, compute_cwc (data1, gama1, gama2, eta, muy, alpha1, alpha2)])

    res = res[res[:, -3] > muy]
    idx = np.argmin (res[:, -1])
    return res[idx].reshape (-1, 9)

res_arr = np.empty ((0, 9))

# Valdation step 
data1, data2 = read_data ()
for eta in tqdm ([1, 5, 10]):
    for muy in [0.8]:#, 0.85, 0.9, 0.95, 0.99]:
        res = determine_pred_interval (data1, eta, muy)
        #print (res)
        res_arr = np.concatenate ((res_arr, res), axis=0)
np.savetxt ("./baseline_PIs_0.8.txt", res_arr, fmt="%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.2f\t%.7f")
