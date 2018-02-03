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
    def __init__(self, dataset_size, dataset_excel_file, k_fold, label, car_ident_flag):
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
        dtype_dict = full_features_dict

        """ These below parameters used in Validation"""
        self.data_training_percentage = 0.8
        self.data_validation_percentage = 0.0
        self.data_test_percentage = 0.2

        # Determine some features
        if car_ident_flag == 0:
            self.features_need_encoding = ["manufacture_code","rep_model_code","car_code","model_code","rating_code","car_type", "trans_mode", "fuel_type", "city", "district", "dealer_name"]
        else:
            self.features_need_encoding = ["car_type", "trans_mode", "fuel_type", "city", "district", "dealer_name"]

        feature_need_label = ["car_type", "trans_mode", "fuel_type", "city", "district", "dealer_name"]
        feature_need_impute = ["rating_code"]
        feature_need_remove_outlier = ["vehicle_mile", "no_click", "recovery_fee"]#, "price"]

        # list of features whether it needs remove outliers 
        feature_need_not_remove_outlier = [feature for feature in self.features if feature not in feature_need_remove_outlier] 
        self.car_ident = ["manufacture_code","rep_model_code","car_code","model_code","rating_code"]
        features_remove_car_ident = [feature for feature in self.features if feature not in self.car_ident] 

        # list of features whether it needs one-hot encode
        self.features_not_need_encoding = [feature for feature in features_remove_car_ident if feature not in self.features_need_encoding] 

        dataframe_file = "./Dataframe/[" + dataset_size + "]total_dataframe_Initial.h5"
        key = "df"

        # These below vars are used in minmax scaler for price
        self.min_price = 1.0
        self.max_price = 100.0

        if os.path.isfile (dataframe_file) == False:
            print ("Load dataset from excel file")
            total_dataset = pd.read_excel (dataset_excel_file, names = self.headers, converters = dtype_dict, header = 0)
            # Shuffle dataset (dataframe)
            #total_dataset = total_dataset.reindex(np.random.permutation(total_dataset.index))
            
            # Remove the data points with sale_state == "advertising"
            total_dataset = total_dataset[total_dataset["sale_state"] == "Sold-out"]
            print ("1.", total_dataset.shape)

            # Remove the data points with cylinder_displayment >=10000 
            total_dataset = total_dataset[total_dataset["cylinder_disp"] < 10000]
            print ("2.1", total_dataset.shape)

            # Remove the data points with  vehicle_mile >=1,000,000,000
            total_dataset = total_dataset[total_dataset["vehicle_mile"] < 1000000000]
            print ("2.2", total_dataset.shape)

            # Remove the data points with maker_year < 2000, > 2018 
            total_dataset = total_dataset[(total_dataset["year"] > 2000) & (total_dataset["year"] < 2018)]
            print ("3.", total_dataset.shape)

            # Remove the data points with price == 0
            #total_dataset = total_dataset[total_dataset["price"] != 0]
            #print ("4.", total_dataset.shape)
            total_dataset = total_dataset[total_dataset["price"] >= 50] # 400]
            print ("3.1", total_dataset.shape)
            #total_dataset = total_dataset[total_dataset["price"] < 9000]
            #print ("3.2", total_dataset.shape)

            # Remove outliers
            #total_dataset = total_dataset[np.abs(total_dataset["price"] - total_dataset["price"].mean()) / total_dataset["price"].std() < 1]
            #print ("4.", total_dataset.shape)

            # Just keep hyundai and kia
            total_dataset = total_dataset[(total_dataset["manufacture_code"] == 101) | (total_dataset["manufacture_code"] == 102)]
            print ("5.", total_dataset.shape)

            # Just keep passenger cars
            total_dataset = total_dataset[(total_dataset["car_type"] == "Passenger car")]
            print ("6.", total_dataset.shape)

            # Replace missing rating_code with 0
            total_dataset["rating_code"] = total_dataset["rating_code"].fillna (0)

            # Remove items with missing actual_adv_date or missing sale_date 
            total_dataset = total_dataset[total_dataset["actual_advertising_date"].notnull() & total_dataset["sale_date"].notnull()]
            print ("7.", total_dataset.shape)

            # Remove the data points with sale duration <= 0
            if label == "sale_duration":
                diff_date = total_dataset["sale_date"]-total_dataset["actual_advertising_date"]
                total_dataset = total_dataset[diff_date > 0] 
                print ("8.", total_dataset.shape)

            # After removing all likely-outliers data points or keep sample items
            # Sort by actual advertising date
            total_dataset = total_dataset.sort_values ("actual_advertising_date", ascending=True)
            
            # use the information about advertising date: use year and month separately
            """adv_date = total_dataset ["actual_advertising_date"]
            manufacture_year = total_dataset ["year"]
            adv_year = adv_date // 10000
            total_dataset["adv_month"] = adv_date % 10000 // 100
            total_dataset["year_diff"] = adv_year - manufacture_year"""

            # Impute missing values from here
            total_dataset = DataFrameImputer().fit_transform (total_dataset)


            # There are some columns with string values (E.g. Car type) -> need to label it as numerical labels
            total_dataset = MultiColumnLabelEncoder(columns = feature_need_label).fit_transform(total_dataset)

            # Standard scale dataset
            #scaler = StandardScaler()  
            scaler = RobustScaler()
            total_dataset[self.features_not_need_encoding] = scaler.fit_transform (total_dataset[self.features_not_need_encoding])

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
   
        #print ("Before scale: min price:", self.min_price, "max price:", self.max_price)
        print ("Time for Loading and preprocessing dataset: %.3f" % (time.time() - stime))
        self.total_dataset = total_dataset
        self.k_fold = k_fold
        
        if car_ident_flag == 1:
            (self.act_adv_date_total_set, self.car_ident_code_total_set, self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set, self.d_ident, self.d_remain, self.car_ident_code_test_set) = self.get_data_label_car_ident (self.features, label)
        else:
            (self.X_total_set, self.y_total_set, self.X_train_set, self.y_train_set, self.X_test_set, self.y_test_set) = self.get_data_label (self.features, label)
    
    
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
            Count how many cars (each car can be diffirentiated by (model_code, rating_code)) in the dataset. 
            Return: 
                + No. different cars in dataset
                + A dictionary that translates the indexs of all cars to the identification numbers .
                + The list of all identification numbers of all cars.
            !NOTE: This method is used to embed car identifications to vectors that are similar if have similar price. (different with create_car_ident() function, -> similar car_ident -> same vector)
        """
        model_code_arr = self.get_data_array (dataset, "model_code")
        print ("model_code_arr", model_code_arr[:5])
        rating_code_arr = self.get_data_array (dataset, "rating_code")
        #print ("rating_code_arr", rating_code_arr)
        
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
                model_rating_dict[model_code_arr[i][0]] = [rating_code_arr[i][0]]
            elif rating_code_arr[i][0] not in model_rating_dict[model_code_arr[i][0]]:
                count += 1
                car_ident_dict[i] = count
                model_rating_dict[model_code_arr[i][0]] += [rating_code_arr[i][0]]
            car_ident_dict[i] = count
            car_ident_list.append (count)
        return (count, car_ident_dict, car_ident_list)

    def create_car_ident (self, dataset):
        """
            From model_code and rating_code (because car identification can be only realized on these 2 features), create another feature: car_ident = 1.1 * model_code + rating_code
            (2 same cars will have the same identification number)
            Return: car_ident
        """
        model_code_arr = self.get_data_array (dataset, "model_code")
        print ("car model_code shape:", model_code_arr.shape)

        rating_code_arr = self.get_data_array (dataset, "rating_code")
        print ("car rating_code shape:", rating_code_arr.shape)
        rating_code_arr = rating_code_arr.astype (int)

        return model_code_arr ** 2 + rating_code_arr


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
    
    def get_sale_duration_array (self, dataset):
        """
        - Purpose: as Return
        - Input: 
            + dataset: training, validation, test, or total dataset
        - Return: an vector oof sale duration as a numpy.array object
        """
        actual_advertising_date_array = np.array ([dataset ["actual_advertising_date"]]).T 
        sale_date_array = np.array ([dataset ["sale_date"]]).T 
        
        length = len (sale_date_array)
        
        sale_duration_array = np.empty((length, 1))
        for i in range (length):
            sale_duration_array[i] = self.get_days_between (actual_advertising_date_array[i][0], sale_date_array[i][0]) 
        
        return sale_duration_array

    def get_data_matrix (self, dataset, features):
        """
        dataset: training, validation, test, or total dataset
        features: an array contains name of features 
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """ 
        X1 = np.array (dataset[self.car_ident + self.features_need_encoding])#["adv_month"] +  
        enc = OneHotEncoder(sparse = False)
        X1 = enc.fit_transform (X1)
        print ("X1.shape", X1.shape)

        X2 = np.array (dataset[self.features_not_need_encoding])#["year_diff"] + 
        X = np.concatenate ((X2, X1), axis = 1) 
        print ("X2.shape", X2.shape)
        return X 

    def get_data_matrix_car_ident (self, dataset):
        """
        dataset: training, validation, test, or total dataset
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """
        # Encode onehot for car identification
        car_ident_codes = np.array (dataset[self.car_ident]) 
        print ("car_ident_codes", car_ident_codes.shape)
        X1 = self.encode_one_hot_car_ident_full (dataset)
        d_ident = X1.shape[1]

        # Encode onehot for the remaining categorical features + maker, rep_model codes
        X2 = np.array (dataset[["manufacture_code","rep_model_code"] + self.features_need_encoding]) 
        enc = OneHotEncoder(sparse = False)
        X2 = enc.fit_transform (X2)

        # Get the remaining features: numerical features
        X3 = np.array (dataset[self.features_not_need_encoding]) 

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
    
    def get_data_label (self, features, label):
        """
            - Purpose: Devide total dataset into n_splits set.
                Using (n_splits - 1) sets for training, the remaining for testing
            - Input:
             + features: an array of feature names
             + label: name of the label (label)
            - Return: a tuple with each element contains a list of "n_splits" training sets (X_train_set, y_train_set) 
             and "n_splits" test sets (X_test_set, y_test_set)
        """
        
        X_total_set = self.get_data_matrix (self.total_dataset, features) 

        len_total_set = X_total_set.shape[0]    
        train_length     = int (0.5 + len_total_set * self.data_training_percentage)

        if label == "price":
            y_total_set = self.get_data_array (self.total_dataset, label)
        elif label == "sale_duration":
            y_total_set = self.get_sale_duration_array (self.total_dataset)
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

            
        return (X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set) 
    
    
    def get_data_label_car_ident (self, features, label):
        """
            - Purpose: Devide total dataset into train and test dataset, or K-fold set.
                Using (n_splits - 1) sets for training, the remaining for testing
            - Input:
             + features: an array of feature names
             + label: name of the label (label)
            - Return: 
                + If use CV: a tuple with each element contains a list of "n_splits" training sets (X_train_set, y_train_set) 
                and "n_splits" test sets (X_test_set, y_test_set)
                + else: return numpy arrays: train, test, ...
        """
        
        car_ident_code_total_set, X_total_set, d_ident, d_remain = self.get_data_matrix_car_ident (self.get_total_dataset ()) 
        act_adv_date = self.get_data_array (self.get_total_dataset (), "actual_advertising_date")

        len_total_set = X_total_set.shape[0]    
        train_length     = int (0.5 + len_total_set * self.data_training_percentage)

        if label == "price":
            y_total_set = self.get_data_array (self.total_dataset, label)
        elif label == "sale_duration":
            y_total_set = self.get_sale_duration_array (self.total_dataset)
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
            
        return (act_adv_date, car_ident_code_total_set, X_total_set, y_total_set, X_train_set, y_train_set, X_test_set, y_test_set, d_ident, d_remain, car_ident_code_test_set) 

    def __str__(self):
        """
        Display a string representation of the data.
        """
        return_str = 'Total dataset'
        return (return_str)




