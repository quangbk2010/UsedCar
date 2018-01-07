import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = 'gb_NN') #"bagging_NN", "gb_NN"
parser.add_argument('--nn_type', type=str, default = 'car2vect') # "baseline", "car2vect"
parser.add_argument('--new_folder', type=str, default = '"saved_model_remove_price_smaller_200_bigger_9000_sortby_adv_date__car2vect_6000_1000_sampleRatio_50"') # name of new folder

args = parser.parse_args()

model = args.model 
nn_type = args.nn_type
new_folder = args.new_folder
num_regressors = 10

if model != "":
    path = "~/UsedCar/checkpoint/" + model + "/" + nn_type + "/regressor"
else:
    path = "~/UsedCar/checkpoint/" + nn_type + "/regressor"
    

for i in range (num_regressors):
    bash_cmd = "cd " + path + str (i+1) + ";"
    bash_cmd += "cp -r temp_save " + new_folder
    os.system (bash_cmd)
