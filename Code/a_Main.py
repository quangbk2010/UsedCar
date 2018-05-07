# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:00:00 2017

@author: quangbk2010


    ########################
    ### MAIN FUNCTION 
    ########################
"""
from a_Dataset import *
from a_Models import *

if __name__ == '__main__':

    device_list = device_lib.list_local_devices()
    print ("\n\n\n============================")
    print ("=== Device list")
    for x in device_list:
        print (x.name)
    print ("============================")

    stime = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--outliers_removal_percent', type=float, default=0.0) # the percentage of data sample in the total dataset used in bagging ensemble method
    parser.add_argument('--test_uncertainty', type=bool, default=False) #"sale_duration" #"price"

    # Configuration
    parser.add_argument('--gpu_idx', type=str, default='0')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--saved_period', type=int, default=10, help='save checkpoint per X epoch') 
    parser.add_argument('--dataframe_dir', type=str, default='../aDataframe/')
    parser.add_argument('--data_dir', type=str, default='../aData/')
    parser.add_argument('--res_dir', type=str, default='../aResult/')
    parser.add_argument('--model_dir', type=str, default='../aSavedModel/')
    parser.add_argument('--train_data_file', type=str, default='Used_car_preprocess1_final.xlsx') #Train_data.xlsx
    parser.add_argument('--test_data_file', type=str, default='Test_data.xlsx')
    parser.add_argument('--test_remain_file', type=str, default='Test_remain.xls')
    parser.add_argument('--test_ident_file', type=str, default='Test_ident.xls')
    parser.add_argument('--test_label_file', type=str, default='Test_label.xls')
    parser.add_argument('--train_rate', type=float, default=0.8)
    parser.add_argument('--valid_rate', type=float, default=0.1)
    parser.add_argument('--best_epoch', type=int, default=49, help='The best epoch corresponding to the best model with smallest error on validation set') 
    parser.add_argument('--restored_epoch', type=int, default=4, help="The epoch corresponding to the saved model for removing outliers") 

    # Hyper parameter
    parser.add_argument('--epoch1', type=int, default=20) 
    parser.add_argument('--epoch2', type=int, default=50) 
    parser.add_argument('--dropout', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.00125)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--decay_step', type=int, default=5) #if decay_step > epoch, no exponential decay

    # Network parameter
    parser.add_argument('--d_embed', type=int, default=3)
    parser.add_argument('--no_hidden_layer', type=int, default=1) 
    parser.add_argument('--no_neuron', type=int, default=6000)
    parser.add_argument('--no_neuron_embed', type=int, default=6000)
    parser.add_argument('--loss_func', type=str, default="rel_err")  
    ######################## end configuration for DL

    args = parser.parse_args()
    if use_gpu == True:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    if not os.path.exists(args.dataframe_dir):
        os.makedirs(args.dataframe_dir)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    ###############################################
    # Determine where to save the results files
    ###############################################
    train_excel_file = args.data_dir + args.train_data_file
    test_data_remain_file = args.data_dir + args.test_remain_file
    test_data_ident_file = args.data_dir + args.test_ident_file
    test_label_file = args.data_dir + args.test_label_file

    # Result file name
    textfile_result = args.res_dir + "All results.txt" 
    dotfile_name_h  = args.res_dir + " Best basic tree_"

    # Files to write the results 
    mean_error_file_name = args.res_dir + "Mean error.txt" 
    y_predict_file_name  = args.res_dir + "[Valid] Predicted price.txt" # store predicted value of validation data
    res_file_name        = args.res_dir + "[Test] Predicted price.txt" # store predicted value of test data

    # Files to write tracking information
    x_embed_file_name   = args.res_dir + "Y embeded.txt" # store the vector after applying car2vect
    x_ident_file_name   = args.res_dir + "X identification.txt" # store the vector after applying car2vect

    if args.mode != "conv_graph":
        print ("\n\n\n##################################")
        print ("# Get the total set")
        print ("##################################")
        dataset     = Dataset (train_excel_file, "train", "", "", "")
        total_data  = dataset.X
        total_label = dataset.y
        d_ident     = dataset.d_ident
        d_remain    = dataset.d_remain

        car_ident_code = dataset.car_ident_code
        act_adv_date   = dataset.act_adv_date
        sale_date      = dataset.sale_date

        if args.mode != "rm_outlier":
            print ("\n\n\n##################################")
            print ("# Get the train, validation set")
            print ("##################################")
            enc1   = dataset.enc1_
            enc2   = dataset.enc2_
            scaler = dataset.scaler

            len_dataset = len (total_label)
            len_train   = int (0.5 + len_dataset * args.train_rate)
            len_vald    = int (0.5 + len_dataset * (args.train_rate+args.valid_rate))
            
            train_data  = total_data  [:len_train] 
            train_label = total_label [:len_train]
            vald_data   = total_data  [len_train:len_vald]
            vald_label  = total_label [len_train:len_vald]

            tv_car_ident_code = car_ident_code[:len_vald]
            tv_act_adv_date   = act_adv_date[:len_vald]
            tv_sale_date      = sale_date[:len_vald]


            print ("train_data: {0}".format (train_data.shape))
            print ("train_label: {0}".format (train_label.shape))
            print ("vald_data: {0}".format (vald_data.shape))
            print ("vald_label: {0}".format (vald_label.shape))


            print ("\n\n\n##################################")
            print ("# Get the test set")
            print ("##################################")

            ###############################################################################
            ## Need to have the test seperated with train and validation set (enc, scaler)
            ###############################################################################

            test_data  = total_data [len_vald:]
            test_label = total_label [len_vald:]
            print ("test_data: {0}".format (test_data.shape))
            print ("test_label: {0}".format (test_label.shape))

            """# Save test data into 2 files, test label into 1 file, that will be used by java source
            test_data_remain = test_data [0:1, 0:d_remain]
            test_data_ident  = test_data [:5, d_remain:]
            df = pd.DataFrame(test_data_remain)
            df.to_excel (test_data_remain_file, index=False)
            sys.exit (-1)
            pd.DataFrame(test_data_ident).to_excel (test_data_ident_file, index=False)
            pd.DataFrame(test_label[:5]).to_excel (test_label_file, index=False)
            sys.exit (-1)"""


            """test_excel_file = args.data_dir + args.test_data_file
            test_dataset    = Dataset (test_excel_file, "test", enc1, enc2, scaler)
            test_data       = test_dataset.X
            test_label      = test_dataset.y 
            
            test_car_ident_code = test_dataset.car_ident_code
            test_act_adv_date   = test_dataset.act_adv_date
            test_sale_date      = test_dataset.sale_date

            print ("test_data:{0}".format (test_data.shape))
            print ("test_label:{0}".format (test_label.shape))

            ############################################
            ## Note: Should remove outliers on testset
            ############################################"""


    print ("\n\n\n##################################")
    print ("# Train and test")
    print ("##################################")
    nn = Tensor_NN (args)

    if args.mode == "rm_outlier":
        print ("=== Remove outlier phase")
        nn.remove_outliers_total_set_car2vec (total_data, total_label, car_ident_code, act_adv_date, sale_date, d_ident, d_remain, y_predict_file_name, args.outliers_removal_percent, args.restored_epoch)

    elif args.mode == "train":
        print ("=== Train phase")
        model_path = nn.model_dir + "final_car2vec_{0}x1_{1}x1".format (nn.no_neuron_embed, nn.no_neuron)
        best_epoch = nn.train_valid_car2vec (train_data, train_label, vald_data, vald_label, tv_car_ident_code, tv_act_adv_date, tv_sale_date, d_ident, nn.d_embed, d_remain, nn.no_neuron, nn.no_neuron_embed, nn.loss_func, model_path, y_predict_file_name, mean_error_file_name, x_ident_file_name, x_embed_file_name, 0, nn.epoch2) # -1 if already have the trained model
        print ("Best epoch:", best_epoch)

    elif args.mode == "test":
        print ("=== Test phase")
        model_path = nn.model_dir + "final_car2vec_{0}x1_{1}x1".format (nn.no_neuron_embed, nn.no_neuron)
        meta_file  = model_path + "_" + str (args.best_epoch) + ".meta"
        ckpt_file  = model_path + "_" + str (args.best_epoch)
        nn.test_car2vec (test_data, test_label, d_ident, d_remain, res_file_name, meta_file, ckpt_file)

    elif args.mode == "conv_graph":
        print ("=== Convert graph phase")
        model_path        = nn.model_dir + "final_car2vec_{0}x1_{1}x1".format (nn.no_neuron_embed, nn.no_neuron)
        meta_file         = model_path + "_" + str (args.best_epoch) + ".meta"
        ckpt_file         = model_path + "_" + str (args.best_epoch)
        frozen_graph_file = ckpt_file + ".pb"
        nn.convert_graph_type (meta_file, ckpt_file, frozen_graph_file)

    else:
        raise ValueError ("This model is not supported!")
    running_time = time.time() - stime
    print ("Time for running: %.3f" % (running_time))
     
