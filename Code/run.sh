#!/bin/bash

rm ../Results/Neural\ network/full\ results/With\ onehot/No\ constraint\ [All\ data]/Validation/predicted*
rm ../Results/Neural\ network/full\ results/With\ onehot/No\ constraint\ [All\ data]/Validation/Mean*

python Neural_network_tensorflow.py --no_neuron 10000 --no_hidden_layer 1 --epoch 20 --scale_label 1

