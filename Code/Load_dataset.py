# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:26:44 2017

@author: quang
"""

from Used_car_regression import *


total_dataset = Dataset()
training_dataset = total_dataset.get_training_dataset()
validation_dataset = total_dataset.get_validation_dataset()

with open ("D:/Projects/Car Price Estimation Project/Results/Simple case/Sale Duration.txt", "w") as f:
    sale_duration_array = total_dataset.get_sale_duration_array (total_dataset.get_total_dataset())
    length = len (sale_duration_array)
            
    for i in range (length):
        if sale_duration_array[i] > 0:
            f.write ("%d\n" % (sale_duration_array[i]))