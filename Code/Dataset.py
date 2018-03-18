# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:00:00 2017

@author: quang

    ########################
    ### PREPARE DATASET
    ########################
    - Input: a csv file that contains data of used cars dataset: 
        + data points: need to extract fo useful features
        + labels: price, sale duration
    - Output: given a new data point-a car (with its data inputs), need to estimate price of this car and how long does this take to sell it.

"""

from Header import *

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

class MultiColumnOutlierRemove:
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
                #output[col] = LabelEncoder().fit_transform(output[col])
                #print ("column:", output[col])
                aColumn = output[col]
                print ("feature:", col, np.array (aColumn).shape, aColumn.mean(), aColumn.std())
                output[col] = output[aColumn < 100] #((aColumn - aColumn.mean()) / aColumn.std()).abs() < 0] #output[np.abs(aColumn - aColumn.mean()) / aColumn.std() < 3)]#.all(axis=1)]
               
                print ("after feature:", col, np.array (output[col]).shape)
 
                #output[col] = output[np.abs (stats.zscore(output[col])) < 3]
        else:
            for colname,col in output.iteritems():
                output[colname] =output[((aColumn - aColumn.mean()) / aColumn.std()).abs() < 3]
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

class Dataset ():
    def __init__(self, dataset_size, dataset_excel_file, k_fold, label, car_ident_flag, get_feature_importance_flag):
        """
            Loading data from csv file
        """
        stime = time.time()
        self.headers = full_features
        if label == "price":
            self.features = features
        elif label == "sale_duration":
            self.features = features + ["price"]

        print ("================length of features:", len (self.features), len (features))
        #dtype_dict = full_features_dict

        """ These below parameters used in Validation"""
        self.data_training_percentage = 0.8
        self.data_validation_percentage = 0.0
        self.data_test_percentage = 0.2

        # Decide to test the affects of price on SD prediction or not
        test_price_SD = False
        test_knn      = False #Only need to set this flag to decide whether use KNN or not 
        self.test_sales_month_effect = False #Only need to set this flag to predict which month to sell a car


        # Determine some features
        if car_ident_flag == 0:
            if dataset_type == "old":
                self.features_need_encoding = ["maker_code","class_code","car_code","model_code","grade_code","car_type", "trans_mode", "fuel_type", "city", "district", "dealer_name"]
            else:
                self.features_need_encoding = ["maker_code","class_code","car_code","model_code","grade_code","car_type","trans_mode","fuel_type","branch","affiliate_code","region","trading_complex","refund","vain_effort","guarantee","selected_color","reg_month","adv_month"]
                if test_knn == True:
                    #self.features_need_encoding = ["class_code","car_code","model_code","car_type"]# NOTE Try with KNN for price prediction
                    self.features_need_encoding = ["class_code","car_type","trading_complex","model_code"]# NOTE Try with KNN for sales duration prediction
                
        else:
            if dataset_type == "old":
                self.features_need_encoding = ["car_type", "trans_mode", "fuel_type", "city", "district", "dealer_name","adv_month"]
            else:
                #self.features_need_encoding = ["car_type","trans_mode","fuel_type","branch","affiliate_code","region","trading_complex"]# NOTE Try to use the same features as old dataset
                if label == "price":
                    self.features_need_encoding = ["car_type","trans_mode","fuel_type","branch","affiliate_code","region","trading_complex","refund","vain_effort","guarantee","selected_color","reg_month","adv_month"] # Price prediction
                else:
                    self.features_need_encoding = ["car_type","trans_mode","fuel_type","branch","affiliate_code","region","trading_complex","refund","vain_effort","guarantee","selected_color","reg_month","adv_month","seller_id","trading_firm_id"] # Sales prediction

        # Add more features to the dataset (before encoding it) to validate the effects of adv_month
        if self.test_sales_month_effect == True:
            self.n_month = 12
            self.new_feature_adv_month = []
            self.new_feature_year_diff = []
            #self.new_feature_adv_month = ["adv_month_1", "adv_month_2", "adv_month_3", "adv_month_4", "adv_month_5", "adv_month_6"]
            #self.new_feature_year_diff = ["year_diff_1", "year_diff_2", "year_diff_3", "year_diff_4", "year_diff_5", "year_diff_6"]
            for i in range (self.n_month):
                self.new_feature_adv_month += ["adv_month_" + str (i + 1)]
                self.new_feature_year_diff += ["year_diff_" + str (i + 1)]
            self.features_need_encoding += self.new_feature_adv_month

        self.features_need_encoding = [] 

        if dataset_type == "old":
            feature_need_label = ["car_type", "trans_mode", "fuel_type", "city", "district", "dealer_name"]
            #self.feature_need_scaled = ["year","vehicle_mile","cylinder_disp", "views", "recovery_fee"]#, "price"] # or = self.features_not_need_encoding
        else:
            #feature_need_label = ["car_type", "trans_mode", "fuel_type", "branch", "region","trading_complex"]# NOTE Try to use the same features as old dataset
            #feature_need_label = ["car_type", "trans_mode", "fuel_type", "branch", "region","trading_complex","refund","vain_effort","guarantee","selected_color"]
            feature_need_label = ["car_type", "trans_mode", "fuel_type", "branch", "region","trading_complex","refund","vain_effort","guarantee","selected_color","seller_id","trading_firm_id"] # Sales prediction
            #self.feature_need_scaled = ["year","vehicle_mile","cylinder_disp","recovery_fee","min_price","max_price","views","no_message_contact","no_call_contact","no_cover_side_recovery","no_cover_side_exchange","no_corrosive_part","no_structure_exchange","mortgage","tax_unpaid","interest"]
        
        feature_need_impute = ["grade_code"]
        self.car_ident = ["maker_code","class_code","car_code","model_code","grade_code"]
        #self.car_ident = ["maker_code","class_code","car_code","grade_code"] # NOTE: Concatenate model_code, grade_code -> grade_code

        # list of features whether it needs remove outliers 
        #feature_need_not_remove_outlier = [feature for feature in self.features if feature not in feature_need_remove_outlier] 
        features_remove_car_ident = [feature for feature in self.features if feature not in self.car_ident] 

        # list of features whether it needs one-hot encode
        
        if car_ident_flag == 0:
            self.features_not_need_encoding = [feature for feature in self.features if feature not in self.features_need_encoding] 
        else:
            self.features_not_need_encoding = [feature for feature in features_remove_car_ident if feature not in self.features_need_encoding] 

        self.feature_need_scaled = self.features_not_need_encoding[:]

        if test_price_SD == True:
            self.feature_need_scaled.remove ("price") # NOTE: Test the affects of price on sales duration

        print ("Length of features need encoding, no need, need scale coresponding are:", len (self.features_need_encoding), len (self.features_not_need_encoding), len (self.feature_need_scaled))
        
        dataframe_file = "./Dataframe/[" + dataset_size + "]total_dataframe_Initial.h5"
        dataframe_file_2 = "./Dataframe/[" + dataset_size + "]total_dataframe_2.h5"
        key = "df"

        # These below vars are used in minmax scaler for price
        self.min_price = 1.0
        self.max_price = 100.0

        if os.path.isfile (dataframe_file) == False:
            print ("Load dataset from excel file")
            total_dataset = pd.read_excel (dataset_excel_file, names = self.headers)#, converters = dtype_dict, header = 0, encoding='utf-8')
            # Shuffle dataset (dataframe)
            #total_dataset = total_dataset.reindex(np.random.permutation(total_dataset.index))
            
            # Remove the data points with sale_state == "advertising"
            if dataset_type == "old":
                total_dataset = total_dataset[total_dataset["sale_state"] == "Sold-out"]
                print ("1.", total_dataset.shape)

            # Remove the data points with cylinder_displayment >=10000 
            total_dataset = total_dataset[total_dataset["cylinder_disp"] < 10000]
            print ("2.1", total_dataset.shape)

            # Remove the data points with vehicle_mile >=1,000,000,000 and < 0
            total_dataset = total_dataset[total_dataset["vehicle_mile"].between (0, 1000000000, inclusive=True)]
            print ("2.2", total_dataset.shape)

            # Remove the data points with revovery_fee < 0
            total_dataset = total_dataset[total_dataset["recovery_fee"] >= 0]
            print ("2.3", total_dataset.shape)

            # Remove the data points with maker_year < 2000, > 2018 
            total_dataset = total_dataset[total_dataset["year"].between (2000, 2018, inclusive=True)] #(total_dataset["year"] > 2000) & (total_dataset["year"] < 2018)]
            print ("3.", total_dataset.shape)

            # Remove the data points with price == 0
            total_dataset = total_dataset[total_dataset["price"] != 0]
            # Remove the items with price is not numerical
            total_dataset = total_dataset[total_dataset[["price"]].applymap (np.isreal).all (1)]
            print ("4.", total_dataset.shape)

            #total_dataset = total_dataset[total_dataset["price"] >= 200] # 400]
            #print ("4.1", total_dataset.shape)
            #total_dataset = total_dataset[total_dataset["price"] < 9000]
            #print ("4.2", total_dataset.shape)

            # Remove outliers
            #total_dataset = total_dataset[np.abs(total_dataset["price"] - total_dataset["price"].mean()) / total_dataset["price"].std() < 1]
            #print ("4.3", total_dataset.shape)
            if label == "sale_duration":
                sales_duration = self.get_sale_duration_array (total_dataset)
                total_dataset ["sale_duration"] = sales_duration

                total_dataset = total_dataset[np.abs(total_dataset["sale_duration"] - total_dataset["sale_duration"].mean()) / total_dataset["sale_duration"].std() < 1]
                print ("4.4", total_dataset.shape)

            # Just keep hyundai and kia
            total_dataset = total_dataset[(total_dataset["maker_code"] == 101) | (total_dataset["maker_code"] == 102)]
            print ("5.", total_dataset.shape)

            # Just keep10 most popular class_code 
            #total_dataset = total_dataset[total_dataset["class_code"].isin ([1101, 1108, 1109, 1166, 1121, 1153, 1225, 1207, 1124, 1151])]
            #print ("5.", total_dataset.shape)

            # Just keep passenger cars
            #total_dataset = total_dataset[(total_dataset["car_type"] == "Passenger car")]
            #print ("6.", total_dataset.shape)

            # Replace missing grade_code with 0
            total_dataset["grade_code"] = total_dataset["grade_code"].fillna (0)

            # Drop all the data points with missing values
            total_dataset = total_dataset.dropna ()
            print ("7.", total_dataset.shape)

            # Concatenate model_code with grade_code to create a new grade_code
            #total_dataset["grade_code"] = total_dataset["model_code"].astype ("int").astype ("str") + total_dataset["grade_code"].astype ("int").astype ("str")

            # Remove the items with missing actual_adv_date or missing sale_date 
            #total_dataset = total_dataset[total_dataset["actual_advertising_date"].notnull() & total_dataset["sale_date"].notnull()]
            #print ("7.", total_dataset.shape)

            # Remove the data points with sale duration <= 0
            if label == "sale_duration":
                if dataset_type == "old":
                    diff_date = total_dataset["sale_date"]-total_dataset["actual_advertising_date"]
                else:
                    diff_date = total_dataset["sale_date"]-total_dataset["first_adv_date"]
                total_dataset = total_dataset[diff_date > 0] 
                print ("8.", total_dataset.shape)

            # After removing all likely-outliers data points or keep sample items
            # Sort by actual advertising date
            if dataset_type == "old":
                total_dataset = total_dataset.sort_values ("actual_advertising_date", ascending=True)
            else:
                total_dataset = total_dataset.sort_values ("first_adv_date", ascending=True)
            
            # use the information about advertising date: use year and month separately
            if dataset_type == "old":
                adv_date = total_dataset ["actual_advertising_date"]
                manufacture_year = total_dataset ["year"]
                adv_year = adv_date // 10000
                total_dataset["adv_month"] = adv_date % 10000 // 100
                total_dataset["year_diff"] = adv_year - manufacture_year

            # Use the information about the first registration date
            #if dataset_type == "new":
            else:
                print ("===Add 4 more features: reg_year, reg_month, adv_month, year_diff!!")
                reg_date = total_dataset ["first_registration"]
                reg_year = reg_date // 10000

                # Remove the data points with maker_year < 2000, > 2018 
                total_dataset = total_dataset[reg_year.between (2000, 2018, inclusive=True)] 
                print ("9.", total_dataset.shape)

                total_dataset["reg_year"] = reg_year
                total_dataset["reg_month"] = reg_date % 10000 // 100

                first_adv_date = total_dataset["first_adv_date"]
                total_dataset["adv_month"] = first_adv_date % 10000 // 100
                first_adv_year = first_adv_date // 10000
                total_dataset["year_diff"] = first_adv_year - reg_year 
                total_dataset["day_diff"]  = self.get_array_days_between (total_dataset, "first_registration", "first_adv_date")


            # Impute missing values from here
            total_dataset = DataFrameImputer().fit_transform (total_dataset)

            # There are some columns with string values (E.g. Car type) -> need to label it as numerical labels
            total_dataset = MultiColumnLabelEncoder(columns = feature_need_label).fit_transform(total_dataset)

            # Standard scale dataset
            #scaler = StandardScaler()  
            #scaler = RobustScaler()
            #total_dataset[self.feature_need_scaled] = scaler.fit_transform (total_dataset[self.feature_need_scaled])

            # MinMax scale the price
            # TODO: here we scale on the total dataset, but we need to scale separately on the train set and the test set
            #scaler = MinMaxScaler(feature_range=(self.min_price, self.max_price))
            #total_dataset["price"] = scaler.fit_transform (total_dataset["price"])

            print ("Store the dataframe into a hdf file")
            total_dataset.to_hdf (dataframe_file, key)       
            print ("Time for Loading dataset: %.3f" % (time.time() - stime))        
        else:
            print ("Reload dataset using HDF5 (Pytables)")
            total_dataset = pd.read_hdf (dataframe_file, key)
   
        # Shuffle dataset (dataframe)
        #total_dataset = total_dataset.reindex(np.random.permutation(total_dataset.index))

        #for feature in features:
        #    print (total_dataset[feature][:3])

        if os.path.isfile (dataframe_file_2) == False:
            print ("Load dataframe_inital from HDF5 file to create dataframe_2")
            if self.test_sales_month_effect == True:
                for i in range (self.n_month):
                    total_dataset[self.new_feature_adv_month [i]] = total_dataset["adv_month"] + i + 1 

                for i in range (self.n_month):
                    total_dataset[self.new_feature_year_diff [i]] = total_dataset["year_diff"] 

                    total_dataset[self.new_feature_year_diff [i]] = total_dataset.apply (
                        lambda row: row[self.new_feature_year_diff [i]]+1 if row[self.new_feature_adv_month [i]]>12 else row[self.new_feature_year_diff [i]],
                        axis=1
                    )
                    total_dataset[self.new_feature_adv_month [i]] = total_dataset.apply (
                        lambda row: row[self.new_feature_adv_month [i]]%12 if row[self.new_feature_adv_month [i]]>12 else row[self.new_feature_adv_month [i]],
                        axis=1
                    )
                self.feature_need_scaled += self.new_feature_year_diff

            # Try to normalize after add more features like year_diff_i due to the effect of year_diff
            scaler = RobustScaler()
            total_dataset[self.feature_need_scaled] = scaler.fit_transform (total_dataset[self.feature_need_scaled])
            print ("Store the dataframe_2 into a hdf file")
            total_dataset.to_hdf (dataframe_file_2, key)       
        else:
            print ("Reload dataframe_file_2 using HDF5 (Pytables)")
            total_dataset = pd.read_hdf (dataframe_file_2, key)

        """test_df = total_dataset [["adv_month"] + self.new_feature_adv_month + ["year_diff"] + self.new_feature_year_diff][:5]
        print (test_df)
        sys.exit (-1)"""

        print ("====Final length of features:", len (self.features))
        #print ("Before scale: min price:", self.min_price, "max price:", self.max_price)
        # Save the length of each vector after encoding into a dictionary
        l_feature = []
        l1 = len (self.features_need_encoding)
        l2 = len (self.features_not_need_encoding)
        # Features names are ordered coresponding to the order of the numpy array columns
        self.sorted_features = self.features_need_encoding + self.features_not_need_encoding
        for i in range (l1):
            l_feature.append (len (np.unique (total_dataset [self.features_need_encoding [i]])))
        for i in range (l2):
            l_feature.append (1)

        if car_ident_flag == 1:
            self.sorted_features = ["maker_code_","class_code_"] + self.sorted_features 
            self.sorted_features += self.car_ident
            l3 = len (self.car_ident)
            for i in range (l3):
                l_feature.append (len (np.unique (total_dataset [self.car_ident [i]])))
                if i == 0:
                    maker_len = l_feature[-1]
                elif i == 1:
                    class_len = l_feature[-1]
            l_feature = [maker_len, class_len] + l_feature


        self.l_feature = l_feature
        if car_ident_flag == 1:
            print ("l1: {0}, l2: {1}, l3: {2}".format (l1, l2, l3))
            print ("l1: {0}, l2: {1}, l3: {2}".format (self.features_need_encoding, self.features_not_need_encoding, self.car_ident))
        else:
            print ("l1: {0}, l2: {1}".format (l1, l2))
            print ("l1: {0}, l2: {1}".format (self.features_need_encoding, self.features_not_need_encoding))
        print ("l_feature: len:", len (l_feature))
        print ("l_feature:", l_feature)

        ############################
        ###NOTE: Test the affects of price on sales duration
        if test_price_SD == True:
            #Test change price by p%
            p = -5.0
            len_total_set = len (total_dataset)    
            len_train  = int (0.5 + len_total_set * self.data_training_percentage)
            total_dataset ["price"][len_train:] *= (1 + p/100.0)
            scaler = RobustScaler()
            total_dataset["price"] = scaler.fit_transform (total_dataset["price"])
        ############################

        rm_idx_file  = "./test_rm_idx.txt"
        outlier_file = "./test_outlier_items.xlsx"
        if os.path.isfile (rm_idx_file) == True:
            # Remove outliers that the coresponding indexes are saved in a file
            with open (rm_idx_file) as f:
                rm_idx = f.readlines()

            rm_idx = [x.strip() for x in rm_idx]
            rm_idx = list (map (int, rm_idx))

            # Save the outliers into a file through a dataframe
            outlier_df = total_dataset.iloc [rm_idx]
            outlier_df[["plate_num", "first_adv_date"]].to_excel (outlier_file, index=False)#, encoding='utf-8-sig')

            print ("######## Length of dataset before removing outlier:", len (total_dataset))
            total_dataset = total_dataset.drop (total_dataset.index [rm_idx])
            print ("######## Length of dataset after removing outlier:", len (total_dataset))

        if car_ident_flag == 1: 
            if dataset_type == "new":
                self.act_adv_date = np.array (total_dataset ["first_adv_date"] ).reshape ((-1,1))
            else:
                self.act_adv_date = np.array (total_dataset ["actual_advertising_date"] ).reshape ((-1,1))
            self.sale_date = np.array (total_dataset ["sale_date"]).reshape ((-1,1))

        print ("Time for Loading and preprocessing dataset: %.3f" % (time.time() - stime))
        self.total_dataset = total_dataset
        self.k_fold = k_fold
       
        if car_ident_flag == 1:
            if self.test_sales_month_effect == True:
                (self.act_adv_date_total_set, self.car_ident_code_total_set, self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set, self.d_ident, self.d_remain, self.car_ident_code_test_set, self.list_test_X) = self.get_data_label_car_ident_2 (label)
            else:
                (self.act_adv_date_total_set, self.car_ident_code_total_set, self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set, self.d_ident, self.d_remain, self.car_ident_code_test_set) = self.get_data_label_car_ident (label, self.features_need_encoding, [])
        else:
            if self.test_sales_month_effect == True:
                (self.act_adv_date_total_set, self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set, self.X_test_set_1, self.X_test_set_2, self.X_test_set_3, self.X_test_set_4, self.X_test_set_5, self.X_test_set_6) = self.get_data_label_2 (label)
            else:
                (self.act_adv_date_total_set, self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set) = self.get_data_label (label, self.features_need_encoding)

        #print ("Dataframe: ", total_dataset ["price"])
        #print ("X[-1]: ", self.X_total_set[:, -1])
        #print (total_dataset ["sale_duration"][:5])
        #print (self.y_total_set[:5])

        #print (min (self.y_total_set)) 
        #print (max (self.y_total_set)) 
        #sys.exit (-1)
        
    
    def get_total_dataset (self):
        return self.total_dataset
    
    def encode_one_hot_feature (self, total_data_array, feature, feature_array):
        """
        There are some features need to be encoded (here using one-hot), because they are categorical features (represented as nominal, is not meaningful because they are index, it doesn't make sense to add or subtract them).
        
        feature_array: numpy array of feature 
        return the one-hot code for feature array
        """
        enc = OneHotEncoder(sparse = False)
        enc.fit (total_data_array.T) 
        return enc.transform (feature_array.T)
        
    def count_car (self, dataset):    
        """
            Count how many cars (each car can be diffirentiated by (model_code, grade_code)) in the dataset. 
            Return: 
                + No. different cars in dataset
                + A dictionary that translates the indexs of all cars to the identification numbers .
                + The list of all identification numbers of all cars.
            !NOTE: This method is used to embed car identifications to vectors that are similar if have similar price. (different with create_car_ident() function, -> similar car_ident -> same vector)
        """
        model_code_arr = self.get_data_array (dataset, "model_code")
        print ("model_code_arr", model_code_arr[:5])
        grade_code_arr = self.get_data_array (dataset, "grade_code")
        #print ("grade_code_arr", grade_code_arr)
        
        len_dataset = model_code_arr.shape[0]
        model_code_list = [] # store all model codes
        car_ident_dict = {} # eg. car_ident[0] = 0, car_ident[1] = 0
        model_rating_dict = {} # eg. model_rating_dict[11045] = [692, 693]
        count = 0
        car_ident_list = []
        
        for i in range (len_dataset):
            if model_code_arr[i][0] not in model_code_list:
                count += 1
                model_code_list.append (model_code_arr[i][0])
                #car_ident[i] = count
                model_rating_dict[model_code_arr[i][0]] = [grade_code_arr[i][0]]
            elif grade_code_arr[i][0] not in model_rating_dict[model_code_arr[i][0]]:
                count += 1
                car_ident_dict[i] = count
                model_rating_dict[model_code_arr[i][0]] += [grade_code_arr[i][0]]
            car_ident_dict[i] = count
            car_ident_list.append (count)
        return (count, car_ident_dict, car_ident_list)

    def create_car_ident (self, dataset):
        """
            From model_code and grade_code (because car identification can be only realized on these 2 features), create another feature: car_ident = 1.1 * model_code + grade_code
            (2 same cars will have the same identification number)
            Return: car_ident
        """
        model_code_arr = self.get_data_array (dataset, "model_code")
        print ("car model_code shape:", model_code_arr.shape)

        grade_code_arr = self.get_data_array (dataset, "grade_code")
        print ("car grade_code shape:", grade_code_arr.shape)
        grade_code_arr = grade_code_arr.astype (int)

        return model_code_arr ** 2 + grade_code_arr


    def encode_one_hot_car_ident_full (self, dataset):
        """ 
            Encode onehot each feature in car-ident, and concatenate these 5 codes into 1 vector, and use this to embed to x_embed with fewer dimension
        """
        enc = OneHotEncoder(sparse = False)
        return enc.fit_transform (np.array (dataset[self.car_ident]))

    def encode_one_hot_car_ident (self, dataset):
        (count, car_ident_dict, car_ident_list) = self.count_car (dataset)
        #car_ident = self.create_car_ident (dataset)
        #car_ident_list = list (car_ident.reshape (car_ident.shape[0]))
        #print ("len of car_ident_list:", len (car_ident_list))
        #print ("no different car identification:", len (Counter(car_ident_list).keys()))

        print ("count:", count, "car_ident_list:", car_ident_list[:50])
        #sys.exit (-1)
        enc = OneHotEncoder(sparse = False)
        return enc.fit_transform (np.array (car_ident_list).reshape (len (car_ident_list), 1)) 
        #return enc.fit_transform (car_ident) 

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
            #print (str_d1, str_d2)
    
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

    def get_sale_duration_array (self, dataset):
        """
        - Purpose: as Return
        - Input: 
            + dataset: training, validation, test, or total dataset
        - Return: an vector oof sale duration as a numpy.array object
        - TODO: can use get_array_days_between() function for this function
        """
        if dataset_type == "old":
            actual_advertising_date_array = np.array ([dataset ["actual_advertising_date"]]).T 
        else:
            actual_advertising_date_array = np.array ([dataset ["first_adv_date"]]).T 
        sale_date_array = np.array ([dataset ["sale_date"]]).T 
        
        length = len (sale_date_array)
        
        sale_duration_array = np.empty((length, 1))
        for i in range (length):
            sale_duration_array[i] = self.get_days_between (actual_advertising_date_array[i][0], sale_date_array[i][0]) 
        
        return sale_duration_array

    def get_data_matrix (self, dataset, features1):
        """
        dataset: training, validation, test, or total dataset
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """ 
        """print (features1)
        X1 = np.array (dataset[features1])  
        enc = OneHotEncoder(sparse = False)
        X1 = enc.fit_transform (X1)
        print ("X1.shape", X1.shape)""" ##NOTE to remove the comments
        
        print (self.features_not_need_encoding)
        X2 = np.array (dataset[self.features_not_need_encoding]) 
        X = X2 #np.concatenate ((X1, X2), axis = 1) 
        print ("X2.shape", X2.shape)


        return X

    def get_onehot_total_set (self):
        X = np.array (self.total_dataset [self.car_ident + self.features_need_encoding])  
        enc = OneHotEncoder (sparse = False)
        enc.fit (X)
        return enc
        
    def get_data_matrix_2 (self, enc, dataset):
        """
        => return: 
            + a matrix with rows are data points, columns are features values (nD numpy.array object)
            + the result encoder
        
        """ 
        X1 = np.array (dataset[self.car_ident + self.features_need_encoding])#["adv_month"] +  
        X1 = enc.transform (X1)
        print ("X1.shape", X1.shape)

        X2 = np.array (dataset[self.features_not_need_encoding])#["year_diff"] + 
        X = np.concatenate ((X2, X1), axis = 1) 
        print ("X2.shape", X2.shape)
        return X

    def get_data_matrix_car_ident (self, dataset, features1):
        """
        dataset: training, validation, test, or total dataset
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """
        print ("get_data_matrix_car_ident", features1)
        # Encode onehot for car identification
        car_ident_codes = np.array (dataset[self.car_ident]) 
        print ("car_ident_codes", car_ident_codes.shape)
        X1 = self.encode_one_hot_car_ident_full (dataset)
        d_ident = X1.shape[1]

        # Encode onehot for the remaining categorical features + maker, rep_model codes
        X3 = np.array (dataset[["maker_code","class_code"] + features1]) 
        enc = OneHotEncoder(sparse = False)
        X3 = enc.fit_transform (X3)

        # Get the remaining features: numerical features
        X2 = np.array (dataset[self.features_not_need_encoding]) 

        # Concatenate 
        X = np.concatenate ((X3, X2, X1), axis = 1) 
        
        d_remain = X.shape[1] - d_ident
        print ("[get_data_matrix_car_ident] shape of prepared data:", X.shape, "d_remain", d_remain, "d_ident", d_ident)
        return (car_ident_codes, X, d_ident, d_remain)

   
    def get_expand_data (self, data):
        """
        Expand data by add ones-vector into the head of data
        """
        one = np.ones ((data.shape[0], 1))
        data_bar = np.concatenate ((one, data), axis = 1)
        return data_bar
    
    def get_data_label (self, label, features1):
        """
            - Purpose: Devide total dataset into n_splits set.
                Using (n_splits - 1) sets for training, the remaining for testing
            - Input:
             + features: an array of feature names
             + label: name of the label (label)
            - Return: a tuple with each element contains a list of "n_splits" training sets (X_train_set, y_train_set) 
             and "n_splits" test sets (X_test_set, y_test_set)
        """
        """if self.test_sales_month_effect == True:
            features1 = list (set (self.features_need_encoding) - set (self.new_feature_adv_month))
        else:
            features1 = self.features_need_encoding"""
        X_total_set = self.get_data_matrix (self.total_dataset, features1) 
        if dataset_type == "old":
            act_adv_date = self.get_data_array (self.get_total_dataset (), "actual_advertising_date")
        else:
            act_adv_date = self.get_data_array (self.get_total_dataset (), "first_adv_date")

        len_total_set = X_total_set.shape[0]    
        train_length     = int (0.5 + len_total_set * self.data_training_percentage)

        """if label == "price":
            y_total_set = self.get_data_array (self.total_dataset, label)
        elif label == "sale_duration":
            y_total_set = self.get_sale_duration_array (self.total_dataset)"""
        y_total_set = self.get_data_array (self.total_dataset, label)

        X_train_set = []
        y_train_set = []
        X_test_set = []
        y_test_set = []
        
        if self.k_fold > 0:
            kf = KFold(self.k_fold, shuffle=True)
            
            for train_index, test_index in kf.split(X_total_set):
                
                X_train, X_test = X_total_set[train_index], X_total_set[test_index]
                y_train, y_test = y_total_set[train_index], y_total_set[test_index]
               
                X_train_set.append (X_train)
                y_train_set.append (y_train)
                X_test_set.append (X_test)
                y_test_set.append (y_test)

        else:
            X_train_set = X_total_set[:train_length, :]
            y_train_set = y_total_set[:train_length, :]

            X_test_set = X_total_set[train_length:, :]
            y_test_set = y_total_set[train_length:, :]

            
        return (act_adv_date, X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set) 

    def get_data_label_2 (self, label):
        """
            - Purpose: Devide total dataset into n_splits set.
                Using (n_splits - 1) sets for training, the remaining for testing
            - Input:
             + label: name of the label (label)
            - Return: Beyond the purpose of get_data_label, this function returns n_month X_test_set_i that the adv_month and adv_year was changed for see the affect of month on price 
        """
        if self.test_sales_month_effect == True:
            features1 = list (set (self.features_need_encoding) - set (self.new_feature_adv_month))
        else:
            features1 = self.features_need_encoding

        (act_adv_date, X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set) = self.get_data_label (label)
        len_total_set = X_total_set.shape[0]    
        train_length     = int (0.5 + len_total_set * self.data_training_percentage)

        return_tuple = (act_adv_date, X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set)

        if self.test_sales_month_effect == True:
            for i in range (self.n_month):
                print ("=====", i)
                features1[2] = self.new_feature_adv_month[i]
                X_total_set_i = self.get_data_matrix (self.total_dataset, features1) 
                X_test_set_i = X_total_set_i[train_length:, :]
                return_tuple.append (X_test_set_i)

        return return_tuple
    
    
    def get_data_label_car_ident (self, label, features1):
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
        
        """if self.test_sales_month_effect == True:
            features1 = list (set (self.features_need_encoding) - set (self.new_feature_adv_month))
        else:
            features1 = self.features_need_encoding"""
        car_ident_code_total_set, X_total_set, d_ident, d_remain = self.get_data_matrix_car_ident (self.get_total_dataset (), features1)

        if dataset_type == "old":
            act_adv_date = self.get_data_array (self.get_total_dataset (), "actual_advertising_date")
        else:
            act_adv_date = self.get_data_array (self.get_total_dataset (), "first_adv_date")

        len_total_set = X_total_set.shape[0]    
        train_length     = int (0.5 + len_total_set * self.data_training_percentage)

        if label == "price":
            y_total_set = self.get_data_array (self.total_dataset, label)
        elif label == "sale_duration":
            y_total_set = self.get_sale_duration_array (self.total_dataset)

            # NOTE: Try to use other units: week, month instead of day
            #y_total_set /= 7
            # Save price, sale_duration
            """price = self.get_data_array (self.get_total_dataset (), "price")
            actual_advertising_date = self.get_data_array (self.get_total_dataset (), "actual_advertising_date")
            sale_date = self.get_data_array (self.get_total_dataset (), "sale_date")
            np_arr = np.concatenate ((price, y_total_set, actual_advertising_date, sale_date), axis=1)
            df = pd.DataFrame (np_arr)
            np.savetxt ("../Results/Price_SaleDuration.txt", df, fmt="%d\t%d\t%s\t%s") 
            sys.exit (-1)"""
            
        X_train_set = []
        y_train_set = []
        X_test_set = []
        y_test_set = []
        car_ident_code_test_set = []

        if self.k_fold > 0:
            kf = KFold(self.k_fold, shuffle=True)
            
            for train_index, test_index in kf.split(X_total_set):
                
                X_train, X_test = X_total_set[train_index], X_total_set[test_index]
                y_train, y_test = y_total_set[train_index], y_total_set[test_index]
                test_car_ident_codes = car_ident_code_total_set[test_index]
               
                X_train_set.append (X_train)
                y_train_set.append (y_train)
                X_test_set.append (X_test)
                y_test_set.append (y_test)
                car_ident_code_test_set.append (test_car_ident_codes)

        else:
            X_train_set = X_total_set[:train_length, :]
            y_train_set = y_total_set[:train_length, :]

            X_test_set = X_total_set[train_length:, :]
            y_test_set = y_total_set[train_length:, :]
            car_ident_code_test_set = car_ident_code_total_set[train_length:, :]
        """print ("######## ", len (car_ident_code_total_set))
        print ("######## ", len (act_adv_date))
        print ("######## ", len (X_total_set))
        sys.exit (-1)"""
            
        return (act_adv_date, car_ident_code_total_set, X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set, d_ident, d_remain, car_ident_code_test_set) 

    def get_data_label_car_ident_2 (self, label):
        """
            - Purpose: Devide total dataset into n_splits set.
                Using (n_splits - 1) sets for training, the remaining for testing
            - Input:
             + label: name of the label (label)
            - Return: Beyond the purpose of get_data_label, this function returns n_month X_test_set_i that the adv_month and adv_year was changed for see the affect of month on price 
        """
        if self.test_sales_month_effect == True:
            features1 = list (set (self.features_need_encoding) - set (self.new_feature_adv_month))
        else:
            features1 = self.features_need_encoding
        len_features= len (features1)
        features1_copy = features1[:]

        (act_adv_date, car_ident_code_total_set, X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set, d_ident, d_remain, car_ident_code_test_set) = self.get_data_label_car_ident (label, features1)
        len_total_set = X_total_set.shape[0]    
        train_length     = int (0.5 + len_total_set * self.data_training_percentage)

        return_tuple = (act_adv_date, car_ident_code_total_set, X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set, d_ident, d_remain, car_ident_code_test_set)

        if self.test_sales_month_effect == True:
            list_test_X = [X_test_set]
            for i in range (self.n_month):
                print ("=====", i)
                for j in range (len_features):
                    if features1_copy[j] == "adv_month":
                        features1[j] = self.new_feature_adv_month[i]
                _, X_total_set_i, _, _  = self.get_data_matrix_car_ident (self.total_dataset, features1) 
                X_test_set_i = X_total_set_i[train_length:, :]
                list_test_X += [X_test_set_i]
            return_tuple += (list_test_X,)

        return return_tuple
    def get_dataset_feature_importance (self, len_train, label):
        """
            Purpose: Split the total data into: train data and a list of test data.
                    A list of test data contains:
                        + The original test data
                        + The test data with each column is permuted, while the other columns keep the initial orders.
        """
        # Separate the total set dataframe into trainset and testset
        train_set = self.total_dataset [:len_train]
        test_set = self.total_dataset [len_train:]

        # Return the onehot encoder with total data, and use it for train and test separately
        enc = self.get_onehot_total_set ()

        # Encode train data and test data
        X_train = self.get_data_matrix_2 (enc, train_set)
        X_test = self.get_data_matrix_2 (enc, test_set)

        #print ("Train shape.", X_train.shape)
        #print ("Test shape.", X_test.shape)

        # Save the incoded data into a list
        list_X_test = [X_test]

        # Permute the testset by each column
        for feature in self.features:
            test_set_copy = test_set.copy ()
            #print ("1. feature: ", test_set[:1])
            test_set_copy [feature] = np.random.permutation (test_set_copy[feature])
            #print ("2. feature: ", test_set[:1])
            X_test = self.get_data_matrix_2 (enc, test_set_copy)
            list_X_test.append (X_test)

        # Get the array of labels
        if label == "price":
            y_train = self.get_data_array (train_set, label)
            y_test = self.get_data_array (test_set, label)
        elif label == "sale_duration":
            y_train = self.get_sale_duration_array (train_set)
            y_test = self.get_sale_duration_array (test_set)
        return (X_train, y_train, list_X_test, y_test)

    def __str__(self):
        """
        Display a string representation of the data.
        """
        return_str = 'Total dataset'
        return (return_str)




