# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:28:28 2017

@author: quang

    JUST FOR TESTING:
"""

#-------------------------------------------------------------------------------------------------------------------------------------------

"""
    NOTE: Need to run Load_dataset first
"""

#from Load_dataset import total_dataset, training_dataset, validation_dataset
from Plot_lib import *
from Evaluate_lib import get_pearsonr_correlation

features = ["vehicle_mile","sale_history","rental_history","total_accident_no","total_water_accident_no","partial_water_accident_no","accident_history_no","recovery_fee","hits","message_contact_no","call_contact_no", "option_navigation","option_sunLoop","option_smartKey","option_xenonLight","option_heatLineSheet","option_ventilationSheet","option_rearSensor","option_curtainAirbag","cover_side_recovery_no","cover_side_exchange_no","corrosive_part_no"]##"price","rep_model_code","car_code","model_code","rating_code",


# Test Dataset class
def test_dataset_class():
    
    print ('---Get training dataset:')
    print (training_dataset)
    
    print ('---Get validation dataset:')
    #array = total_dataset.get_data_array(total_dataset.get_validation_dataset(), 'rating_name')
    print (validation_dataset)
    
    print ('---Get a matrix of features from total dataset:')
    features = ["manufacture_name","representative_name", "manufacture_code"]
    matrix = total_dataset.get_data_matrix (total_dataset.get_training_dataset(), features)  
    matrix_bar = total_dataset.get_expand_data (matrix)
    print (matrix_bar)
    
    print ('---Get an array of a feature with constraint from total dataset:')
    array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), 'manufacture_code', 'manufacture_name', 'Kia')
    print (array)
    
    print ('---Get an array of a feature from total dataset:')
    array = total_dataset.get_data_array (total_dataset.get_total_dataset(), 'manufacture_code')
    print (array)
#test_dataset_class()

# Test plot functions and check data distribution with each output
def test_plot_function(feature, output):
    data_array = total_dataset.get_data_array (total_dataset.get_total_dataset(), feature)
    label_array = total_dataset.get_data_array (total_dataset.get_total_dataset(), output)
    
    axis_limit (0, 200000, 0, 5000)
    plot_data_label (data_array, label_array, 'Distribution of ' + feature + ' with ' + output,  'g.', feature, output)
    plt.show()
    
    data_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_name', 'Kia')
    label_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), output, 'manufacture_name', 'Kia')
    
    axis_limit (0, 200000, 0, 5000)
    plot_data_label (data_array, label_array, 'Distribution of ' + feature + ' with ' + output + ' and constraint: manufacture is Kia',  'g.', feature, output)
    plt.show()
    
    data_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_name', 'Benz')
    label_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), output, 'manufacture_name', 'Benz')
    
    axis_limit (0, 200000, 0, 5000)
    plot_data_label (data_array, label_array, 'Distribution of ' + feature + ' with ' + output + ' and constraint: manufacture is Benz',  'g.', feature, output)
    plt.show()
    """
    axis_limit (0, 2000, 0, 15000)
    plot_data_label (data_array, label_array, 'Distribution of ' + feature + ' with ' + output,  'g.', feature, output)
    plt.show()
    
    data_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_name', 'Kia')
    label_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), output, 'manufacture_name', 'Kia')
    
    axis_limit (0, 2000, 0, 6000)
    plot_data_label (data_array, label_array, 'Distribution of ' + feature + ' with ' + output + ' and constraint: manufacture is Kia',  'g.', feature, output)
    plt.show()
    
    data_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_name', 'Benz')
    label_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), output, 'manufacture_name', 'Benz')
    
    axis_limit (0, 2000, 0, 15000)
    plot_data_label (data_array, label_array, 'Distribution of ' + feature + ' with ' + output + ' and constraint: manufacture is Benz',  'g.', feature, output)
    plt.show()
    
    data_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_name', 'Hyundai')
    label_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), output, 'manufacture_name', 'Hyundai')
    
    axis_limit (0, 2000, 0, 10000)
    plot_data_label (data_array, label_array, 'Distribution of ' + feature + ' with ' + output + ' and constraint: manufacture is Hyundai',  'g.', feature, output)
    plt.show()
    
    data_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_name', 'BMW')
    label_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), output, 'manufacture_name', 'BMW')
    
    axis_limit (0, 2000, 0, 10000)
    plot_data_label (data_array, label_array, 'Distribution of ' + feature + ' with ' + output + ' and constraint: manufacture is BMW',  'g.', feature, output)
    plt.show()
    
    data_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_name', 'Ssangyong')
    label_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), output, 'manufacture_name', 'Ssangyong')
    
    axis_limit (0, 2000, 0, 5000)
    plot_data_label (data_array, label_array, 'Distribution of ' + feature + ' with ' + output + ' and constraint: manufacture is Ssangyong',  'g.', feature, output)
    plt.show()
    
    data_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_name', 'Lexus')
    label_array = total_dataset.get_data_array_with_constraint (total_dataset.get_total_dataset(), output, 'manufacture_name', 'Lexus')
    
    axis_limit (0, 2000, 0, 9000)
    plot_data_label (data_array, label_array, 'Distribution of ' + feature + ' with ' + output + ' and constraint: manufacture is Lexus',  'g.', feature, output)
    plt.show()
    """
        
#test_plot_function ('vehicle_mile', 'price') 
#test_plot_function ('accident_history_no', 'price')  
#test_plot_function ('hits', 'price') 

def test_plot_cdf():
    for feature in features:
        data_array = total_dataset.get_data_array_without_onehot (total_dataset.get_total_dataset(), feature)
        plot_cdf (feature, data_array, '')

#test_plot_cdf()

def test_plot_cdf_with_constraint():
    manufacture_code_constraint = [189]#101, 102, 103, 105, 107] # coresponds to: Hyundai, Kia, GM Korea, Ssangyong, Renault Samsung, 
    for manufacture_code in manufacture_code_constraint:
        print ('manufacture code:', manufacture_code)
        for feature in features:
            data_array = total_dataset.get_data_array_without_onehot_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_code', manufacture_code)
            #normalized_data_array = data_array / max (data_array)
            plot_cdf (feature, data_array, 'Manufacture code:'+ str(manufacture_code))

#test_plot_cdf_with_constraint()   

def test_plot_cdf_with_constraint_vehicle_mile():
    for feature in features:
        data_array = total_dataset.get_data_array_without_onehot_with_constraint (total_dataset.get_total_dataset(), feature, "vehicle_mile", 200000)
        normalized_data_array = data_array / max (data_array)
        plot_cdf (feature, normalized_data_array, 'distance driven <= 200000')

#test_plot_cdf_with_constraint_vehicle_mile()

def test_plot_cdf_with_constraint_not_polular_car():
    manufacture_code_polular = [101, 102, 103, 105, 107] # coresponds to: Hyundai, Kia, GM Korea, Ssangyong, Renault Samsung
    manufacture_code_not_popular = []
    manufacture_code_array = total_dataset.get_data_array_without_onehot (total_dataset.get_total_dataset(), 'manufacture_code')
    for code in manufacture_code_array:
        if (code not in manufacture_code_polular) and (code not in manufacture_code_not_popular):
            manufacture_code_not_popular.append (code[0])
    print ('Not polular manufacture code:', manufacture_code_not_popular)
    manufacture_code_not_popular_sorted = sorted (manufacture_code_not_popular)
    
    for manufacture_code in manufacture_code_not_popular_sorted:
        print ('manufacture code:', manufacture_code)
        for feature in features:
            data_array = total_dataset.get_data_array_without_onehot_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_code', manufacture_code)
            normalized_data_array = data_array / max (data_array)
            plot_cdf (feature, normalized_data_array, 'Manufacture code:'+ str(manufacture_code))
            
    
#test_plot_cdf_with_constraint_not_polular_car()      
        
def test_correlation():
    #pearsonr_array = []
    price = total_dataset.get_data_array_without_onehot (total_dataset.get_total_dataset(), "price")
    normalized_price = price / max (price)
    for feature in features:
        data_array = total_dataset.get_data_array_without_onehot (total_dataset.get_total_dataset(), feature)
        normalized_data_array = data_array / max (data_array)
        corre = get_pearsonr_correlation (normalized_data_array, normalized_price)
        #pearsonr_array.append (corre[0][0])
        print (feature + ' ', corre[0][0])
    #print ('pearsonr_array:', pearsonr_array)
#test_correlation()

def test_correlation_constraint_vehicle_mile():
    #pearsonr_array = []
    price = total_dataset.get_data_array_without_onehot (total_dataset.get_total_dataset(), "price")
    normalized_price = price / max (price)
    for feature in features:
        data_array = total_dataset.get_data_array_without_onehot_with_constraint (total_dataset.get_total_dataset(), feature, "vehicle_mile", 200000)
        normalized_data_array = data_array / max (data_array)
        corre = get_pearsonr_correlation (normalized_data_array, normalized_price)
        #pearsonr_array.append (corre[0][0])
        print (feature + ' ', corre[0][0])
    #print ('pearsonr_array:', pearsonr_array)
test_correlation()

def test_correlation_with_constraint():
    #pearsonr_array = []
    manufacture_code_constraint = [108, 109, 114, 115, 116, 137, 189]#101, 102, 103, 105, 107] # coresponds to: Hyundai, Kia, GM Korea, Ssangyong, Renault Samsung,
    for manufacture_code in manufacture_code_constraint:
        print ('manufacture code:', manufacture_code)
        price = total_dataset.get_data_array_without_onehot_with_constraint (total_dataset.get_total_dataset(), 'price', 'manufacture_code', manufacture_code)
        normalized_price = price / max (price)
        for feature in features:
            data_array = total_dataset.get_data_array_without_onehot_with_constraint (total_dataset.get_total_dataset(), feature, 'manufacture_code', manufacture_code)
            normalized_data_array = data_array / max (data_array)
            corre = get_pearsonr_correlation (normalized_data_array, normalized_price)
            #pearsonr_array.append (corre[0][0])
            print (feature + ' ', corre[0][0])
        print()
    #print ('pearsonr_array:', pearsonr_array)

#test_correlation_with_constraint()

   
    
    
    
    
    