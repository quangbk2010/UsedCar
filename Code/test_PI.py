import numpy as np
import pandas as pd

from datetime import datetime
import sys
np.set_printoptions (threshold = np.nan)
pd.set_option('display.max_columns', None)
import math
import scipy.stats as stats

def compute_cwc (data, gama1, gama2, eta=10, muy=0.95, alpha1=0, alpha2=0):
    len_data = len (data)
    R = np.max (data["price"]) - np.min (data["price"])
    data["lower"]  = data["pred_price"] * (1 - gama1) - alpha1 * data["std"]
    data["higher"] = data["pred_price"] * (1 + gama2) + alpha2 * data["std"]
    mpiw = np.mean (data["pred_price"] * (gama1 + gama2))
    no_in_range    = np.sum ((data["price"] > data["lower"]) & (data["price"] < data["higher"]))
    
    picp = no_in_range / len_data 
    nmpiw = mpiw / R
    #cwc = nmpiw * (1 + gama (picp, muy) * math.exp (-eta * (picp - muy)))
    cwc = nmpiw * math.exp (-eta * (picp - muy))
    #print ("===%.2f, %.2f, %.2f, %.2f, %.2f, %.4f" % ( gama1, gama2, nmpiw, picp, muy, cwc)) # nmpiw is small as possible, percent is large as posible, cwc is small as possible
    
    return (gama1, gama2, eta, alpha1, alpha2, nmpiw, picp, muy, cwc)

def determine_pred_interval (eta=10, percent_threshold=0.95):
    data = pd.read_csv ("../Results/price.txt", sep="\t")
    len_data = len (data)
    len1 = int (len_data / 2)
    #print (data.shape)
    data["delta"] = data["pred_price"] - data["price"]    
    # Divide it into 2 part: validation set and test set
    data1 = data [:len1]
    data2 = data [len1:]
    muy = percent_threshold # eta: 10, 100, 300
    res = np.empty ((0, 9))
    # Determine the range that ground truth included in the range from the validation set   
    for gama1 in np.arange (0.05, 0.25, 0.01):#np.arange (0.05, 0.3, 0.05):
        for gama2 in np.arange (0.05, 0.25, 0.01):#np.arange (0.05, 0.3, 0.05):
            for alpha1 in np.arange (0, 5.1, 0.5):
                for alpha2 in np.arange (0, 5.1, 0.5):
                    res = np.vstack ([res, compute_cwc (data1, gama1, gama2, eta, muy, alpha1, alpha2)])
    res = res[res[:, -3] > muy]
    idx = np.argmin (res[:, -1])
    return res[idx].reshape (-1, 9)

res_arr = np.empty ((0, 9))

# Valdation step 
for eta in [1, 5, 10]:
    for muy in [0.9]:#0.8, 0.85, 0.9, 0.95, 0.99]:
        res = determine_pred_interval (eta, muy)
        #print (res)
        res_arr = np.concatenate ((res_arr, res), axis=0)
np.savetxt ("./baseline_PIs.txt", res_arr, fmt="%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.2f\t%.7f")
