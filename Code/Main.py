# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:00:00 2017

@author: quang


    ########################
    ### MAIN FUNCTION 
    ########################
"""
from Dataset import *
from Models import *

if __name__ == '__main__':
    stime = time.time()
    parser = argparse.ArgumentParser()

    print ("1.", device_lib.list_local_devices())
    parser.add_argument('--label', type=str, default = 'price') #"sale_duration" #"price"
    parser.add_argument('--model_set', type=str, default = 'DL') # "Sklearn" or "DL"
    parser.add_argument('--dataset_size', type=str, default = 'new') # new, "full", or "partial", or "small"
    parser.add_argument('--car_ident_flag', type=int, default = 0) # 1-yes, 0-no
    parser.add_argument('--ensemble_NN_flag', type=int, default = 0) # 0-no, 1-gb_NN_baseline, 2-gb_NN_car2vect, 3-gb_NN_baseline_tree, 4-bagging_NN_baseline, 5-bagging_NN_car2vect, 6-retrain_car2vect_after_remove_outliers, 7-retrain_car2vect_from_total_set, 8-regularization
    parser.add_argument('--num_regressor', type=int, default = 0) # the number of regressors used for gradient boosting
    parser.add_argument('--sample_ratio', type=float, default = 1.0) # the percentage of data sample in the total dataset used in bagging ensemble method
    parser.add_argument('--outliers_removal_percent', type=float, default = 0.0) # the percentage of data sample in the total dataset used in bagging ensemble method
    parser.add_argument('--get_feature_importance_flag', type=bool, default = False) # the flag that decide whether evaluate the features importance or not
    parser.add_argument('--price_change_percent', type=float, default = 0.0) #"sale_duration" #"price"
    parser.add_argument('--test_uncertainty', type=bool, default = False) #"sale_duration" #"price"
    parser.add_argument('--test_kb', type=bool, default = False) #"sale_duration" #"price"



    ########################
    ### Linear models
    ########################
    # Add configuration for Linear models here
    ######################## end configuration for Linear models 

    ########################
    ### Decision Trees
    ########################
    # Add configuration for DT here
    ######################## end configuration for DT

    ########################
    ### Deep Learning
    ########################
    #configuration
    parser.add_argument('--gpu_idx', type=str, default = '0')
    parser.add_argument('--model_dir', type=str, default = '../checkpoint')
    parser.add_argument('--model_name', type=str, default = 'usedCar')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--saved_period', type=int, default=100, help='save checkpoint per X epoch') # 200
    parser.add_argument('--output_dir', type=str, default = '../Results')
    parser.add_argument('--scale_label', type=int, default=0)
    parser.add_argument('--use_gpu', type=bool, default=True)

    #hyper parameter
    parser.add_argument('--epoch', type=int, default = 10) #2000 # 100
    parser.add_argument('--dropout', type=float, default = 1)
    parser.add_argument('--batch_size', type=int, default = 256)#512)
    parser.add_argument('--learning_rate', type=float, default=0.00125)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--decay_step', type=int, default=5) #if decay_step > epoch, no exponential decay

    #network parameter
    parser.add_argument('--d_embed', type=int, default=3)
    parser.add_argument('--no_hidden_layer', type=int, default = 1) 
    parser.add_argument('--no_neuron', type=int, default = 1000)
    parser.add_argument('--no_neuron_embed', type=int, default = 6000)
    parser.add_argument('--k_fold', type=int, default = -1) # set it to -1 when don't want to use k-fold CV
    parser.add_argument('--loss_func', type=str, default = "rel_err") #rel_err, mae, rmse, smape 
    ######################## end configuration for DL

    print ("2.", device_lib.list_local_devices())
    args = parser.parse_args()
    if args.use_gpu == True:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    print ("3.", device_lib.list_local_devices())
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    ###############################################
    # Determine where to save the results files
    ###############################################
    dataset_size = args.dataset_size
    if dataset_size == "full":
        # Dataset length
        input_no = 239444# 239472 #
        dataset_excel_file = '../Data/used_car_Eng_pre_processing1.xlsx'
    elif dataset_size == "partial":
        input_no = 8000
        dataset_excel_file = '../Data/used_car_Eng_pre_processing1_partial dataset.xlsx'
    elif dataset_size == "small":
        input_no = 14
        dataset_excel_file = '../Data/used_car_Eng_pre_processing1_small dataset.xlsx'
    elif dataset_size == "new":
        input_no = 270968
        dataset_excel_file = '../Data/Used_car_preprocess1_final.xlsx'
    elif dataset_size == "new_small":
        input_no = 10
        dataset_excel_file = '../Data/Used_car_preprocess1_final_small.xlsx'
    print ('Data set length:', input_no)
    print ("Label:", args.label)

    # Result file name
    pre_file_name = directory_h + dataset_size + " results/" + feature_coding + "/"
    if using_CV_flag == 1:
        pre_file_name += "Kfold_CV/"
    else:
        pre_file_name += "Validation/"

    if not os.path.exists(pre_file_name):
        os.makedirs(pre_file_name)
    textfile_result = pre_file_name + "All results.txt" # " Result " + list_regression_model[0] + "_testing.txt"
    dotfile_name_h  = pre_file_name + " Best basic tree_"

    # Files to write the results 
    mean_error_file_name = pre_file_name + "Mean error.txt" # "Mean_rmse over K-fold CV.txt"#" Mean_rmse over K-fold CV_for_testing.txt"
    y_predict_file_name = pre_file_name + "predicted " + args.label + ".txt" # store predicted value

    # Files to write tracking information
    x_embed_file_name = pre_file_name + "Y embeded.txt" # store the vector after applying car2vect
    x_ident_file_name = pre_file_name + "x identification.txt" # store the vector after applying car2vect
    total_set_file_name = pre_file_name + "total set.txt" # store the data set after shuffle total numpy array
    train_set_file_name = pre_file_name + "train set.txt" # store the training set after shuffle total numpy array
    test_set_file_name = pre_file_name + "test set.txt" # store the test set after shuffle total numpy array


    ##################################
    # Get the dataset
    ##################################
    dataset = Dataset (args.dataset_size, dataset_excel_file, args.k_fold, args.label, args.car_ident_flag, args.get_feature_importance_flag, args.ensemble_NN_flag, "", "", "", "")
    total_data  = dataset.X_total_set
    total_label = dataset.y_total_set
    
    train_data  = dataset.X_train_set
    train_label = dataset.y_train_set
    test_data   = dataset.X_test_set
    test_label  = dataset.y_test_set
    total_act_adv_date= dataset.act_adv_date_total_set
    sorted_features = dataset.sorted_features

    """list_test_data = dataset.list_test_X
    if list_test_data[0].shape[1] != list_test_data[1].shape[1]:
        raise ValueError ("Oops! #columns of X_test_set != of X_test_set_1")"""

    total_car_ident_code= dataset.car_ident_code_total_set
    total_act_adv_date = dataset.act_adv_date
    total_sale_date = dataset.sale_date
    test_car_ident = dataset.car_ident_code_total_set[train_data.shape[0]:]

    print ("train_data:", train_data.shape)
    print ("train_label:", train_label.shape)
    print ("test_data:", test_data.shape)
    print ("test_label:", test_label.shape)

    """enc1 = dataset.enc1_
    enc2 = dataset.enc2_
    scaler = dataset.scaler
    #print (total_data[:1,:])

    
    print ("##################################")
    print ("# Get the test_kb set")
    print ("##################################")
    ### Domestic cars
    kb_domestic_dataset_excel_file = '../Data/kb_results_combined_car_type_domestic_v2.xlsx'
    test_kb = True
    kb_domestic_dataset = Dataset (args.dataset_size, kb_domestic_dataset_excel_file, args.k_fold, args.label, args.car_ident_flag, args.get_feature_importance_flag, args.ensemble_NN_flag, "domestic", enc1, enc2, scaler)
    kb_domestic_test_data  = kb_domestic_dataset.X_total_set
    kb_domestic_test_label = np.zeros ((len (kb_domestic_test_data), 1))
    
    kb_domestic_act_adv_date= kb_domestic_dataset.act_adv_date_total_set

    kb_domestic_car_ident_code= kb_domestic_dataset.car_ident_code_total_set
    kb_domestic_act_adv_date = kb_domestic_dataset.act_adv_date
    kb_domestic_sale_date = kb_domestic_dataset.sale_date
    kb_domestic_car_ident = kb_domestic_dataset.car_ident_code_total_set

    print ("kb_domestic_data:", kb_domestic_test_data.shape)
    print ("kb_domestic_label:", kb_domestic_test_label.shape)
    ###############################################################"""

    ### Import cars
    """kb_income_dataset_excel_file = '../Data/kb_results_combined_car_type_import_v2.xlsx'
    kb_income_dataset = Dataset (args.dataset_size, kb_income_dataset_excel_file, args.k_fold, args.label, args.car_ident_flag, args.get_feature_importance_flag, args.ensemble_NN_flag, "import", enc1, enc2, scaler)
    kb_income_test_data  = kb_income_dataset.X_total_set
    kb_income_test_label = np.zeros ((len (kb_income_test_data), 1))
    
    kb_income_act_adv_date= kb_income_dataset.act_adv_date_total_set

    kb_income_car_ident_code= kb_income_dataset.car_ident_code_total_set
    kb_income_act_adv_date = kb_income_dataset.act_adv_date
    kb_income_sale_date = kb_income_dataset.sale_date
    kb_income_car_ident = kb_income_dataset.car_ident_code_total_set

    print ("kb_income_data:", kb_income_test_data.shape)
    print ("kb_income_label:", kb_income_test_label.shape)
    ##################################"""
    # Test feature importance
    """# Test feature importance when using model: NN_baseline
    if args.get_feature_importance_flag == True:
        X_train, y_train, list_X_test, y_test = dataset.X_train, dataset.y_train, dataset.list_X_test, dataset.y_test"""
    l_feature = dataset.l_feature
    ##################################

    ##################################
    # Determine model to predict
    ##################################
    if args.model_set == "sklearn":
        model = Sklearn_model (dataset_size)
        for reg_type in ["DecisionTreeRegressor"]:#"Linear", "Ridge", "Lasso", "DecisionTreeRegressor", "RandomForestRegressor", "AdaBoostRegressor", "GradientBoostingRegressor"]:
            model.sklearn_regression(reg_type, train_data, train_label, test_data, test_label)
        #model.knn (train_data, train_label, test_data, test_label)

    elif args.model_set == "DL":
        nn = Tensor_NN (args)
 
        if args.ensemble_NN_flag == 1:
            nn.gradient_boosting_NN_baseline (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, dataset_size=dataset_size)

        elif args.ensemble_NN_flag == 2:
            nn.gradient_boosting_NN_car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, test_car_ident=test_car_ident, d_ident=dataset.d_ident, d_remain=dataset.d_remain, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, dataset_size=dataset_size, retrain=0)

        elif args.ensemble_NN_flag == 3:
            num_trees = 2
            nn.gradient_boosting_NN_baseline_Tree (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, dataset_size=dataset_size)

        elif args.ensemble_NN_flag == 4:
            nn.bagging_NN_baseline (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, total_car_ident=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, y_predict_file_name=y_predict_file_name, x_ident_file_name=x_ident_file_name, mean_error_file_name=mean_error_file_name, dataset_size=dataset_size, ratio=nn.sample_ratio)

        elif args.ensemble_NN_flag == 5:
            nn.bagging_NN_car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, test_car_ident=test_car_ident, d_ident=dataset.d_ident, d_remain=dataset.d_remain, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, dataset_size=dataset_size, ratio=nn.sample_ratio, retrain=0)

        elif args.ensemble_NN_flag == 6:
            nn.retrain_car2vect_after_remove_outliers (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, test_car_ident=test_car_ident, d_ident=dataset.d_ident, d_remain=dataset.d_remain, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, dataset_size=dataset_size, removal_percent=args.outliers_removal_percent)

        elif args.ensemble_NN_flag == 7:
            nn.retrain_car2vect_from_total_set (total_data=total_data, total_label=total_label, total_car_ident_code=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, d_ident=dataset.d_ident, d_remain=dataset.d_remain, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, dataset_size=dataset_size, removal_percent=args.outliers_removal_percent, ensemble_flag=0, l_feature=l_feature, features=sorted_features)

        elif args.ensemble_NN_flag == 71:
            nn.retrain_car2vect_from_total_set (total_data=total_data, total_label=total_label, total_car_ident_code=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, d_ident=dataset.d_ident, d_remain=dataset.d_remain, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, dataset_size=dataset_size, removal_percent=args.outliers_removal_percent, ensemble_flag=-1, l_feature=l_feature, features=sorted_features) # The 2 last attributes used for calculating features importance

        elif args.ensemble_NN_flag == 72:
            nn.retrain_car2vect_from_total_set (total_data=total_data, total_label=total_label, total_car_ident_code=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, d_ident=dataset.d_ident, d_remain=dataset.d_remain, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, dataset_size=dataset_size, removal_percent=args.outliers_removal_percent, ensemble_flag=2, l_feature=l_feature, features=sorted_features)

        elif args.ensemble_NN_flag == 75:
            nn.retrain_car2vect_from_total_set (total_data=total_data, total_label=total_label, total_car_ident_code=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, d_ident=dataset.d_ident, d_remain=dataset.d_remain, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, dataset_size=dataset_size, removal_percent=args.outliers_removal_percent, ensemble_flag=5, l_feature=l_feature, features=sorted_features)

        elif args.ensemble_NN_flag == 8:
            model_path = nn.model_dir + "/car2vect/[{0}]{1}_{2}_car2vect_{3}_{4}_{5}_{6}".format (dataset_size, nn.model_name, args.label, nn.no_neuron, nn.no_neuron_embed, nn.d_embed, nn.loss_func)
            nn.car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, total_car_ident=total_car_ident_code, d_ident=dataset.d_ident, d_embed=nn.d_embed, d_remain=dataset.d_remain, no_neuron=nn.no_neuron, no_neuron_embed=nn.no_neuron_embed, loss_func=nn.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, retrain=0) 

        elif args.ensemble_NN_flag == 9:
            nn.retrain_baseline_from_total_set (total_data=total_data, total_label=total_label, total_car_ident_code=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, y_predict_file_name=y_predict_file_name, x_ident_file_name=x_ident_file_name, mean_error_file_name=mean_error_file_name, dataset_size=dataset_size, removal_percent=args.outliers_removal_percent, ensemble_flag=0, l_feature=l_feature, features=sorted_features)

        elif args.ensemble_NN_flag == 91:
            nn.retrain_baseline_from_total_set (total_data=total_data, total_label=total_label, total_car_ident_code=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, y_predict_file_name=y_predict_file_name, x_ident_file_name=x_ident_file_name, mean_error_file_name=mean_error_file_name, dataset_size=dataset_size, removal_percent=args.outliers_removal_percent, ensemble_flag=-1, l_feature=l_feature, features=sorted_features)

        elif args.ensemble_NN_flag == 100:
            model_path = nn.model_dir + "/baseline/test" # + "/car2vect/[{0}]{1}_{2}_car2vect_{3}_{4}_{5}_{6}".format (dataset_size, nn.model_name, args.label, nn.no_neuron, nn.no_neuron_embed, nn.d_embed, nn.loss_func)
            if nn.car_ident_flag == 1:
                nn.test_affect_price_sales_duration_car2vec (train_data, train_label, test_data, test_label, dataset.d_ident, dataset.d_remain, model_path, y_predict_file_name)
            else:
                nn.test_affect_price_sales_duration_baseline (train_data, train_label, test_data, test_label, model_path, y_predict_file_name)

        elif args.ensemble_NN_flag == 101:
            model_path = nn.model_dir + "/car2vect/test"
            nn.decide_when_sell_car (list_test_data, test_label, dataset.d_ident, dataset.d_remain, model_path)

        elif args.ensemble_NN_flag == 10:
            model_path = ""
            nn.car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, total_car_ident=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, d_ident=dataset.d_ident, d_embed=nn.d_embed, d_remain=dataset.d_remain, no_neuron=nn.no_neuron, no_neuron_embed=nn.no_neuron_embed, loss_func=nn.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, retrain=-1) 

        elif nn.car_ident_flag == 1:
            model_path = nn.model_dir + "/rm_outliers_total_set_NN/car2vect/regressor{0}/{1}_{2}_{3}_car2vect_{4}x1_{5}x1_total_set".format (2, dataset_size, nn.model_name, nn.label, nn.no_neuron_embed, nn.no_neuron)
            if args.get_feature_importance_flag == True:
                nn.get_features_importance_car2vect (train_data, train_label, test_data, test_label, total_car_ident_code, dataset.d_ident, dataset.d_remain, model_path, sorted_features, l_feature)
            elif args.test_kb == True:
                #best_epoch = nn.car2vect (train_data=[], train_label=[], test_data=kb_domestic_test_data, test_label=kb_domestic_test_label, total_car_ident=kb_domestic_car_ident_code, total_act_adv_date=kb_domestic_act_adv_date, total_sale_date=kb_domestic_sale_date, d_ident=dataset.d_ident, d_embed=nn.d_embed, d_remain=dataset.d_remain, no_neuron=nn.no_neuron, no_neuron_embed=nn.no_neuron_embed, loss_func=nn.loss_func, model_path=model_path, y_predict_file_name="", mean_error_file_name="", x_ident_file_name=x_ident_file_name + "_domestic", x_embed_file_name="", retrain=-1) 
                best_epoch = nn.car2vect (train_data=[], train_label=[], test_data=kb_income_test_data, test_label=kb_income_test_label, total_car_ident=kb_income_car_ident_code, total_act_adv_date=kb_income_act_adv_date, total_sale_date=kb_income_sale_date, d_ident=dataset.d_ident, d_embed=nn.d_embed, d_remain=dataset.d_remain, no_neuron=nn.no_neuron, no_neuron_embed=nn.no_neuron_embed, loss_func=nn.loss_func, model_path=model_path, y_predict_file_name="", mean_error_file_name="", x_ident_file_name=x_ident_file_name + "_import", x_embed_file_name="", retrain=-1) 
            elif args.test_uncertainty == True:
                best_epoch = nn.car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, total_car_ident=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, d_ident=dataset.d_ident, d_embed=nn.d_embed, d_remain=dataset.d_remain, no_neuron=nn.no_neuron, no_neuron_embed=nn.no_neuron_embed, loss_func=nn.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, retrain=-1) 
            else:
                print ("4.", device_lib.list_local_devices())
                best_epoch = nn.car2vect (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, total_car_ident=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, d_ident=dataset.d_ident, d_embed=nn.d_embed, d_remain=dataset.d_remain, no_neuron=nn.no_neuron, no_neuron_embed=nn.no_neuron_embed, loss_func=nn.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name, mean_error_file_name=mean_error_file_name, x_ident_file_name=x_ident_file_name, x_embed_file_name=x_embed_file_name, retrain=0) 
                print ("Best epoch: ", best_epoch)
                running_time = time.time() - stime
                print ("Time for running: %.3f" % (running_time))
                line = np.zeros ((0, 4))
                #content=np.array (["use_gpu", "epoch", "batch_size", "running_time"]).reshape (1, 4)
                content = np.array ([args.use_gpu, args.epoch, args.batch_size, running_time]).reshape (1, 4)
                line = np.concatenate ((line,content), axis = 0)
                with open ("./test_time_{0}.txt".format (args.use_gpu), "ab") as file:
                    np.savetxt (file, line, fmt="%s\t%s\t%s\t%s")

        else:
            model_path = nn.model_dir + "/baseline/[{0}]{1}_{2}_baseline_{3}_{4}_{5}".format (dataset_size, nn.model_name, args.label, nn.no_neuron, nn.no_hidden_layer, nn.loss_func)
            if args.get_feature_importance_flag == True:
                nn.get_features_importance_baseline_NN (train_data, train_label, test_data, test_label, model_path, sorted_features, l_feature) 
            else:
                best_epoch = nn.baseline (train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label, total_car_ident=total_car_ident_code, total_act_adv_date=total_act_adv_date, total_sale_date=total_sale_date, no_neuron=nn.no_neuron, no_hidden_layer = nn.no_hidden_layer, loss_func=nn.loss_func, model_path=model_path, y_predict_file_name=y_predict_file_name, x_ident_file_name=x_ident_file_name, mean_error_file_name=mean_error_file_name)

                print ("Best epoch: ", best_epoch)

    else:
        raise ValueError ("This model is not supported!")
     
