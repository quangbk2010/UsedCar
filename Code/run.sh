#!/bin/bash

#######################
#rm ../Results/Neural\ network/full\ results/With\ onehot/No\ constraint\ [All\ data]/Validation/predicted*
#rm ../Results/Neural\ network/full\ results/With\ onehot/No\ constraint\ [All\ data]/Validation/Mean*

#######################
#baseline NN# python Main.py --model_set DL --dataset_size full --ensemble_NN_flag 0 --loss_func rel_err --car_ident_flag 0 --no_neuron 1000 --no_hidden_layer 1 --epoch 30
#car2vect NN# python Main.py --model_set DL --dataset_size full --ensemble_NN_flag 0 --loss_func rel_err --car_ident_flag 1 --no_neuron 1000 --no_neuron_embed 6000 --epoch 30


#######################
#Gradient boosting with baseline NN# python Main.py --model_set DL --dataset_size full --ensemble_NN_flag 1 --loss_func mse --car_ident_flag 0 --no_neuron 1000 --no_hidden_layer 1 --epoch 30 --num_regressor 3
#Gradient boosting with car2vect NN# python Main.py --model_set DL --dataset_size full --ensemble_NN_flag 2 --loss_func mse --car_ident_flag 1 --no_neuron 1000 --no_neuron_embed 6000 --epoch 30 --num_regressor 3


#######################
#Bagging with baseline NN# python Main.py --model_set DL --dataset_size full --ensemble_NN_flag 4 --loss_func mse --car_ident_flag 0 --no_neuron 1000 --no_hidden_layer 2 --epoch 30 --num_regressor 3
#Bagging with baseline NN# python Main.py --model_set DL --dataset_size full --ensemble_NN_flag 5 --loss_func mse --car_ident_flag 1 --no_neuron 1000 ---no_neuron_embed 6000 --epoch 30 --num_regressor 3


#######################
python Main.py --model_set DL --dataset_size full --ensemble_NN_flag 4 --car_ident_flag 0 --no_neuron 1000  --no_hidden_layer 2 --epoch 30 --num_regressor 3 --sample_ratio 0.5

#######################
#banana2
cd ~/UsedCar/Code/Dataframe/
scp \[full\]total_dataframe_Initial.h5 quang@banana1.kaist.ac.kr:~/UsedCar/Code/Dataframe

cd ~/UsedCar/checkpoint/gb_NN/baseline/regressor1
scp checkpoint full_usedCar_price_baseline_2000_2_rel_err_9.* quang@banana4.kaist.ac.kr:~/UsedCar/checkpoint/gb_NN/baseline/regressor1

#######################
#client host
cd ~/UsedCar/checkpoint
mkdir -p gb_NN/baseline
cd gb_NN/baseline
mkdir regressor1 regressor2 regressor3 regressor4 regressor5 regressor6 regressor7 regressor8 regressor9 regressor10 



