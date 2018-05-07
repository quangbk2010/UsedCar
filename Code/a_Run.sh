#!/bin/bash

if [ "$1" == "rm_outlier" ]; then
    if [ "$#" -eq 3 ]; then
        echo "======================================================================================"
        echo "=== If \$3 < 0, then train the model from scratch with the total dataset"
        echo "=== Otherwise, restore the pre-trained model with restored_epoch \$3"
        echo "======================================================================================"
        python a_Main.py --mode rm_outlier --outliers_removal_percent $2 --restored_epoch $3
    elif [ "$#" -eq 2 ]; then
        echo "==================================================================="
        echo "=== Restore the pre-trained model with default restored_epoch=4"
        echo "==================================================================="
        python a_Main.py --mode rm_outlier --outliers_removal_percent $2
    else
        echo -e "\n!!! We need at least 2 arguments: <rm_outlier> [\"restored_epoch\"] <\"outlier_removal_percent\">\n"
        echo "If restored_epoch < 0, then train from scratch."
    fi

elif [ "$1" == "train" ]; then
    python a_Main.py --mode train

elif [ "$1" == "test" ]; then
    if [ "$#" -eq 2 ]; then
        python a_Main.py --mode test --best_epoch $2 
    else
        echo "=================================================="
        echo "=== Use the default model with best_epoch=49"
        echo "=================================================="
        python a_Main.py --mode test
    fi

elif [ "$1" == "conv_graph" ]; then
    if [ "$#" -eq 2 ]; then
        python a_Main.py --mode conv_graph --best_epoch $2
    else
        echo "=================================================="
        echo "=== Use the default model with best_epoch=49"
        echo "=================================================="
        python a_Main.py --mode conv_graph
    fi

else
    echo -e "\nValueError: Unsupported mode!!!"
    echo "There are 3 modes: rm_outlier, train, test"
    echo -e "The first argument should be one of these modes.\n"
fi
