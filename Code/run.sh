#!/bin/bash

#rm ../Results/Neural\ network/full\ results/With\ onehot/No\ constraint\ [All\ data]/Validation/predicted*
#rm ../Results/Neural\ network/full\ results/With\ onehot/No\ constraint\ [All\ data]/Validation/Mean*

#python Neural_network_tensorflow.py --no_neuron 10000 --no_hidden_layer 1 --epoch 20 --scale_label 1
#python Neural_network_tensorflow.py --no_neuron 1000 --no_neuron_embed 1000 --epoch 50 --scale_label 1

#python Main.py --model_set DL --dataset_size full --gb_NN_flag 1 --loss_func mse --car_ident_flag 0 --no_neuron 1000 --no_hidden_layer 1 --epoch 30 --num_regressor 3
#python Main.py --model_set DL --dataset_size full --gb_NN_flag 2 --loss_func mse --car_ident_flag 1 --no_neuron 1000  --no_neuron_embed 6000 --epoch 30 --num_regressor 3
python Main.py --model_set DL --dataset_size full --gb_NN_flag 3 --loss_func mse --car_ident_flag 0 --no_neuron 1000  --no_hidden_layer 2 --epoch 30 --num_regressor 3
