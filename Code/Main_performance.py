# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:28:28 2017

@author: quang

    JUST FOR TESTING:
"""
#-------------------------------------------------------------------------------------------------------------------------------------------
"""
    NOTE: Need to run Load_training first (or any time we change something in class Training)
"""

from Used_car_regression import *  
from Evaluate_lib import *


# Test Training class:
def test_training_class():
    regr = training.linear_regression (features, output)
    print (regr.coef_)
    plot_fitting_line (regr.coef_, 'rv', 1)
    
#test_training_class() 


"""def test_neural_network():
    nn = training.neural_network_regression (50, x, y)
    x_ = np.linspace(-4, 4, 160) # define axis

    pred_x = np.reshape(x_, [160, 1]) # [160, ] -> [160, 1]
    pred_y = nn.predict(pred_x) # predict network output given x_
    fig = plt.figure() 
    plt.plot(x_, x_**2, color = 'b') # plot original function
    plt.plot(pred_x, pred_y, '-') # plot network output
    plt.show ()

test_neural_network()"""




#test = Test_validation ()
#test.test_RMSE_scoring_function ()    

  










