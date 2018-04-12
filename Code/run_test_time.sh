#!/bin/bash

# use_gpu in only set in Header.py file
list_batch_size=(16 64 128 256 512 )
list_epoch=(1 5 )
for epoch in ${list_epoch[@]}
do
    for batch_size in ${list_batch_size[@]}
    do
        echo "=====Test:" $use_gpu $epoch $batch_size
        #python Main.py --model_set DL --dataset_size new --ensemble_NN_flag 0 --no_neuron 6000 --no_neuron_embed 6000 --car_ident_flag 1 --outliers_removal_percent 5 --dropout 0.5 --use_gpu False --epoch 1
        python Main.py --model_set DL --dataset_size new --ensemble_NN_flag 0 --no_neuron 6000 --no_neuron_embed 6000 --car_ident_flag 1 --dropout 0.5 --epoch $epoch --batch_size $batch_size
    done
done
