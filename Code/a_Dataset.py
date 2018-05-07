# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:00:00 2017

@author: quangbk2010

    ########################
    ### PREPARE DATASET
    ########################
    - Input: a csv file that contains data of used cars dataset: 
        + data points: need to extract fo useful features
        + labels: price, sale duration
    - Output: given a new data point-a car (with its data inputs), need to estimate price of this car and how long does this take to sell it.

"""

from a_Header import *

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        # NOTE: the year (eg. advertising date) will be affected if using the mean function
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') or X[c].dtype == np.dtype('int64') else X[c].mean() for c in X], index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)        

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

class Dataset ():
    def __init__(self, dataset_excel_file, data_type, enc1, enc2, scaler):
        """
            Loading data from csv file
        """
        stime = time.time()
        self.headers = full_features
        self.features = features

        """ These below parameters used in Validation"""
        # Decide to test the affects of price on SD prediction or not
        self.test_sales_month_effect = False #Only need to set this flag to predict which month to sell a car


        # Determine some features need one hot encode
        #self.features_need_encoding = ["car_type","trans_mode","fuel_type","branch","affiliate_code","region","trading_complex","refund","vain_effort","guarantee","selected_color","reg_month","adv_month"] 
        self.features_need_encoding = ["car_type","trans_mode","fuel_type","selected_color","adv_month","reg_month"]

        # Add more features to the dataset (before encoding it) to validate the effects of adv_month
        #feature_need_label = ["car_type", "trans_mode", "fuel_type", "branch", "region","trading_complex","refund","vain_effort","guarantee","selected_color","seller_id","trading_firm_id"] 
        feature_need_label = ["car_type", "trans_mode", "fuel_type","selected_color"] 
        
        self.car_ident = ["maker_code","class_code","car_code","model_code","grade_code"]

        # list of features don't include car_ident 
        features_remove_car_ident = [feature for feature in self.features if feature not in self.car_ident] 

        # list of features don't needs one-hot encode
        self.features_not_need_encoding = [feature for feature in features_remove_car_ident if feature not in self.features_need_encoding] 
        self.feature_need_scaled = self.features_not_need_encoding[:]

        dataframe_file_0 = "../aDataframe/total_dataframe_0.h5" + data_type
        dataframe_file_1 = "../aDataframe/total_dataframe_1.h5" + data_type
        dataframe_file_2 = "../aDataframe/total_dataframe_2.h5" + data_type
        key = "df"

        if os.path.isfile (dataframe_file_0) == False:
            print ("Load dataset from excel file")
            total_dataset = pd.read_excel (dataset_excel_file, names = self.headers)#, converters = dtype_dict, header = 0, encoding='utf-8')
            print (total_dataset.shape)
            print ("Store the dataframe into a hdf file")
            total_dataset.to_hdf (dataframe_file_0, key)       
            print ("Time for Loading dataset: %.3f" % (time.time() - stime))        
        else:
            print ("Reload dataset using HDF5 (Pytables)")
            total_dataset = pd.read_hdf (dataframe_file_0, key)

        if os.path.isfile (dataframe_file_1) == False:
            # Remove the data points with cylinder_displayment >=10000 
            total_dataset = total_dataset[total_dataset["cylinder_disp"] < 10000]
            print ("1.{0}".format(total_dataset.shape))

            # Remove the data points with vehicle_mile >=1,000,000,000 and < 0
            total_dataset = total_dataset[total_dataset["vehicle_mile"].between (0, 1000000000, inclusive=True)]
            print ("2.{0}".format(total_dataset.shape))

            # Remove the data points with revovery_fee < 0
            total_dataset = total_dataset[total_dataset["recovery_fee"] >= 0]
            print ("3.{0}".format(total_dataset.shape))

            # Remove the data points with maker_year < 2000, > 2018 
            total_dataset = total_dataset[total_dataset["year"].between (2000, 2018, inclusive=True)] #(total_dataset["year"] > 2000) & (total_dataset["year"] < 2018)]
            print ("4.{0}".format(total_dataset.shape))

            # Remove the data points with price == 0
            total_dataset = total_dataset[total_dataset["price"] != 0]
            # Remove the items with price is not numerical
            total_dataset = total_dataset[total_dataset[["price"]].applymap (np.isreal).all (1)]
            print ("5.{0}".format(total_dataset.shape))

            # Just keep hyundai and kia
            #total_dataset = total_dataset[(total_dataset["maker_code"] == 101) | (total_dataset["maker_code"] == 102)]
            #print ("6.{0}".format(total_dataset.shape))

            # Replace missing grade_code with 0
            total_dataset["grade_code"] = total_dataset["grade_code"].fillna (0)

            # Drop all the data points with missing values
            total_dataset = total_dataset.dropna ()
            print ("7.{0}".format(total_dataset.shape))

            # Sort by actual advertising date
            total_dataset = total_dataset.sort_values ("first_adv_date", ascending=True)
           
            ###############################
            # Use the information about the first registration date
            reg_date = total_dataset ["first_registration"]
            #reg_year = reg_date // 10000

            # Remove the data points with maker_year < 2000, > 2018 
            #total_dataset = total_dataset[reg_year.between (2000, 2018, inclusive=True)] 
            total_dataset = total_dataset[total_dataset["year"].between (2000, 2018, inclusive=True)] 
            print ("8.{0}".format(total_dataset.shape))

            #total_dataset["reg_year"] = reg_year
            total_dataset["reg_month"] = reg_date % 10000 // 100

            first_adv_date = total_dataset["first_adv_date"]
            total_dataset["adv_month"] = first_adv_date % 10000 // 100
            first_adv_year = first_adv_date // 10000
            #total_dataset["year_diff"] = first_adv_year - reg_year 
            total_dataset["year_diff"] = first_adv_year - total_dataset["year"] 
            total_dataset["day_diff"]  = self.get_array_days_between (total_dataset, "first_registration", "first_adv_date")
            ###############################


            # Impute missing values from here
            if data_type == "train":
                total_dataset = DataFrameImputer().fit_transform (total_dataset)
            else:
                total_dataset["grade_code"] = pd.to_numeric(total_dataset["grade_code"], errors='coerce').fillna(0)

            # There are some columns with string values (E.g. Car type) -> need to label it as numerical labels
            total_dataset = MultiColumnLabelEncoder(columns = feature_need_label).fit_transform(total_dataset)

            print ("Store the dataframe into a hdf file")
            total_dataset.to_hdf (dataframe_file_1, key)       
            print ("Time for Loading dataset: %.3f" % (time.time() - stime))        
        else:
            print ("Reload dataset using HDF5 (Pytables)")
            total_dataset = pd.read_hdf (dataframe_file_1, key)

   
        # Only keep the cars with car's age <= 10 years
        total_dataset = total_dataset [total_dataset ["year_diff"] <= 10] 
        print ("9.{0}".format(total_dataset.shape))

        if scaler == "":
            scaler = RobustScaler()
            total_dataset[self.feature_need_scaled] = scaler.fit_transform (total_dataset[self.feature_need_scaled])
        else:
            total_dataset = total_dataset.fillna (0)
            total_dataset[self.feature_need_scaled] = scaler.transform (total_dataset[self.feature_need_scaled])
            
        self.scaler = scaler
        print ("Store the dataframe_2 into a hdf file")

        # Remove outliers that the coresponding indexes are saved in a file
        # Problem: if we remove from the original dataset with this file, the shape of the fitting data will be different from with what used to train the model
        # Solution: we need to train again from the removed dataset
        rm_idx_file  = "./test_rm_idx.txt"
        outlier_file = "./test_outlier_items.xlsx"
        if os.path.isfile (rm_idx_file) == True: 
            with open (rm_idx_file) as f:
                rm_idx = f.readlines()

            rm_idx = [x.strip() for x in rm_idx]
            rm_idx = list (map (int, rm_idx))

            # Save the outliers into a file through a dataframe
            outlier_df = total_dataset.iloc [rm_idx]
            outlier_df[["plate_num", "first_adv_date"]].to_excel (outlier_file, index=False)#, encoding='utf-8-sig')

            print ("--> Length of dataset before removing outlier: {0}".format(len (total_dataset)))
            total_dataset = total_dataset.drop (total_dataset.index [rm_idx])
            print ("--> Length of dataset after removing outlier: {0}".format(len (total_dataset)))

        self.act_adv_date = np.array (total_dataset ["first_adv_date"] ).reshape ((-1,1))
        self.sale_date = np.array (total_dataset ["sale_date"]).reshape ((-1,1))

        print ("Time for loading and preprocessing dataset: %.3f" % (time.time() - stime))

        #################################################
        ### Save prediction results
        """len_total_df = len (total_dataset)
        len_train_df = int (len_total_df * 0.9 + 0.5)
        #print (len_total_df, len_train_df, len_total_df - len_train_df)
        test_data_df = total_dataset[full_features][len_train_df:]
        test_data_df = test_data_df
        #print (test_data_df)

        test_res_df = pd.read_csv ("../aResult/[Test] Predicted price.txt", sep="\t", names=["Price", "Pred_price", "Rel_err"])
        test_res_df = test_res_df
        test_data_df.index = test_res_df.index
        res_df = test_data_df
        for c in ["Price", "Pred_price", "Rel_err"]:
            res_df[c] = test_res_df[c]

        #print (res_df)
        #res_df.to_csv ("../aResult/[Test] Results.csv", sep=",", encoding="utf-8", index=False)
        res_df.to_excel ("../aResult/[Test] Results.xlsx", encoding="utf-8", index=False)
        sys.exit (-1)"""
        #################################################

        self.total_dataset = total_dataset

        (self.car_ident_code, self.X, self.y, self.d_ident, self.d_remain, self.enc1_, self.enc2_) = self.get_data_label_car_ident (self.features_need_encoding, enc1, enc2)

    
    def get_total_dataset (self):
        return self.total_dataset
    
    def get_data_array (self, dataset, feature):  
        """
        dataset: training, validation, test, or total dataset
        feature: is a element in feature list
        return: an vector (1D numpy.array object)
        """
        total_data_array = np.array ([self.get_total_dataset()[feature]])
        data_array = np.array ([dataset[feature]])
        
        return data_array.T
    
    def get_days_between(self, d1, d2):
        if np.isnan(d1) or np.isnan(d2):
            return (-1)
        str_d1 = str (int (d1))
        str_d2 = str (int (d2))
        try:
            d1 = datetime.strptime(str_d1, "%Y%m%d")
            d2 = datetime.strptime(str_d2, "%Y%m%d")
            return int (abs((d2 - d1).days))
        except ValueError:
            # NOTE: prevent error when impute the missing values with mean() or some format mistakes
            return (-1)
    
    def get_array_days_between (self, dataset, name1, name2):
        """
        - Purpose: as Return
        - Input: 
            + dataset: training, validation, test, or total dataset
        - Return: an vector of #days between 2 arrays as a numpy.array object
        """
        array1 = np.array ([dataset [name1]]).T 
        array2 = np.array ([dataset [name2]]).T 
        
        len_data = len (array1)     
        delta_list = [self.get_days_between(array1[i], array2[i]) for i in range (len_data)]
        
        return delta_list

    def get_data_matrix_car_ident (self, dataset, features1, enc1, enc2):
        """
        dataset: training, validation, test, or total dataset
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """
        # Encode onehot for car identification
        car_ident_codes = np.array (dataset[self.car_ident]) 
        X1 = np.array (dataset[self.car_ident])
        if enc1 != "":
            X1 = enc1.transform (X1)
        else:
            enc1 = OneHotEncoder(sparse = False, handle_unknown="ignore")
            X1 = enc1.fit_transform (X1)
        d_ident = X1.shape[1]

        # Encode onehot for the remaining categorical features + maker, rep_model codes
        X3 = np.array (dataset[["maker_code","class_code"] + features1]) 
        if enc2 != "":
            X3 = enc2.transform (X3)
        else:
            enc2 = OneHotEncoder(sparse = False, handle_unknown="ignore")
            X3 = enc2.fit_transform (X3)

        # Get the remaining features: numerical features
        X2 = np.array (dataset[self.features_not_need_encoding]) 

        # Concatenate 
        X = np.concatenate ((X3, X2, X1), axis = 1) 
        
        d_remain = X.shape[1] - d_ident
        #print ("[get_data_matrix_car_ident] shape of prepared data:{0},{1},{2},{3}".format(X.shape, d_remain, d_ident, X3.shape))
        return (car_ident_codes, X, d_ident, d_remain, enc1, enc2)

   
    def get_data_label_car_ident (self, features1, enc1, enc2):
        """
            - Purpose: Devide total dataset into train and test dataset, or K-fold set.
                Using (n_splits - 1) sets for training, the remaining for testing
            - Input:
                + label: name of the label (label)
            - Return: 
                + If use CV: a tuple with each element contains a list of "n_splits" training sets (X_train_set, y_train_set) 
                and "n_splits" test sets (X_test_set, y_test_set)
                + else: return numpy arrays: train, test, ...
        """
        car_ident_code, X, d_ident, d_remain, enc1_, enc2_ = self.get_data_matrix_car_ident (self.get_total_dataset (), features1, enc1, enc2)

        y = self.get_data_array (self.total_dataset, "price")
            
        return (car_ident_code, X, y, d_ident, d_remain, enc1_, enc2_) 

    def __str__(self):
        """
        Display a string representation of the data.
        """
        return_str = 'Total dataset'
        return (return_str)




