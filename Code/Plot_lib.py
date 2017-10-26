# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:32:51 2017

@author: quang

    DRAW FIGURE:
     - Plot data, label distribution for each feature
     - Plot fitting line for different algorithns
"""

from Header_lib import *

def axis_limit (axis_x_min, axis_x_max, axis_y_min, axis_y_max):
    plt.axis ([axis_x_min, axis_x_max, axis_y_min, axis_y_max])
def plot_data_label (X, y, title,  color_shape, x_label, y_label):
    plt.plot (X, y, color_shape)
    plt.title (title)
    plt.xlabel (x_label)
    plt.ylabel (y_label)
    
def plot_fitting_line (w, color, degree=1, start=0, stop=14000, num=140000):
    w_array = []
    for i in range (degree + 1):
        w_array.append (w[0][i])
    x0 = np.linspace (start, stop, num)
    y0 = w_array[0]
    for i in range (1, degree + 1):
        y0 += w_array[i] * (x0 ** i)
    plt.plot (x0, y0, color)
    
def plot_cdf (feature, data_array, constraint):
    data_sorted = sorted (data_array)
    y_vals = np.arange (len (data_sorted)) / float (len (data_sorted))
    #print ('x:', data_sorted)
    #print ('y:', y_vals)
    """if feature == "vehicle_mile":
        axis_limit (0, 500000, 0, 1.0)
    elif feature == 'recovery_fee':
        axis_limit (0, 40000000, 0, 1.0)
    elif feature == 'hits':
        axis_limit (0, 1000, 0, 1.0)
    elif feature == 'price':
        axis_limit (0, 10000, 0, 1.0)"""
    plot_data_label (data_sorted, y_vals, 'CDF (' + constraint + ')',  'b.', feature, 'Cumulative probability')
    plt.plot (data_sorted, y_vals)
    plt.show ()

def plot_tree_all_Kfold ():
    print ("Start plot tree:")
    #system ("dot -Tpng ../Results/Simple case/Tree plot/basic_tree_0.dot -o ../Results/Simple case/Tree plot/basic_tree.png")

    for i in range (K_fold):
        for max_depth in max_depth_list_h:#, 10**5]:       
            for max_leaf_nodes in max_leaf_nodes_list_h:#, 10**5]:
                
                fn = dotfile_name_h + str (max_depth) + "_" + str (max_leaf_nodes) + "_" + str (i)
                
                dotfile = fn + ".dot"
                pngfile = fn + ".png"
                
                #print ("dotfile: ", dotfile)
                
                command = ["dot", "-Tpng", dotfile, "-o", pngfile]
             
                try:
                    subprocess.check_call(command)
                except IOError as e:
                    print ("I/O error({0}): {1}".format(e.errno, e.strerror))
                except ValueError:
                    print ("Value Error!")
                except:
                    print ("Unexpected error!")
        print ("End of plotting %d iteration" % (i))

def plot_tree_final ():
    
    print ("Start plot tree:")
    
    fn = dotfile_name_h + str (max_depth) + "_" + str (max_leaf_nodes)
    
    dotfile = fn + ".dot"
    pngfile = fn + ".png"
    
    #print ("dotfile: ", dotfile)
    
    command = ["dot", "-Tpng", dotfile, "-o", pngfile]
 
    try:
        subprocess.check_call(command)
    except IOError as e:
        print ("I/O error({0}): {1}".format(e.errno, e.strerror))
    except ValueError:
        print ("Value Error!")
    except:
        print ("Unexpected error!")
        
    print ("End of plotting!")
plot_tree_final ()








