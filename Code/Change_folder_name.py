import os

model = "gb_NN" #"bagging_NN"
nn_type = "car2vect" # "baseline"
num_regressors = 10

if model != "":
    path = "~/UsedCar/checkpoint/" + model + "/" + nn_type + "/regressor"
else:
    path = "~/UsedCar/checkpoint/" + nn_type + "/regressor"
    
new_folder = "saved_model_remove_price_smaller_200_bigger_9000_sortby_adv_date__car2vect_6000_1000_sampleRatio_50"

for i in range (num_regressors):
    bash_cmd = "cd " + path + str (i+1) + ";"
    bash_cmd += "cp -r temp_save " + new_folder
    os.system (bash_cmd)
