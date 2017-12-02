# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:09:55 2017

@author: quang

- Input: a csv file that contains data of used cars dataset: 
    + data points: need to extract fo useful features
    + labels: price, sale duration
- Output: given a new data point-a car (with its data inputs), need to estimate price of this car and how long does this take to sell it.

- Implimentation: Divide into 2 categories: LOAD DATA (Dataset class), TRAIN DATA (Training class)

"""

from Header_lib import *
#from Neural_network import *
"""
    LOAD DATA:
     - Load data from csv file, and store as narrays
"""

class Data_preprocessing (object):
    def __init__(self):
        pass

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
            #raise

    def shuffle(self, df, n=1, axis=0):     
        df = df.copy()
        for _ in range(n):
            df.apply(np.random.shuffle, axis=axis)
        return df

    def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

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

class Dataset (Data_preprocessing, DataFrameImputer):
    def __init__(self):
        """
        Loading data from csv file
        skip_rows: the number of row will be skipped from row 0
        skip_footer: the number of row will be skipped from the end
        """
        stime = time.time()
        self.headers = full_features
        dtype_dict = full_features_dict
        total_dataset = pd.read_excel (dataset_excel_file, names = self.headers, converters = dtype_dict, header = 0)
        print ("1.", total_dataset.shape)

        
        # Shuffle dataset (dataframe)
        total_dataset = total_dataset.reindex(np.random.permutation(total_dataset.index))

        
        # Sort by actual advertising date
        #total_dataset = total_dataset.sort_values ("actual_advertising_date", ascending=True)
        
        # Remove the data points with price == 0
        #total_dataset = total_dataset[total_dataset["sale_state"] == "Sold-out"]
        #print ("2.", total_dataset.shape)

        # Remove the data points with price == 0
        #total_dataset = total_dataset[total_dataset["price"] != 0]
        #print ("3.", total_dataset.shape)

        # Remove outliers
        #total_dataset = total_dataset[np.abs(total_dataset["price"] - total_dataset["price"].mean()) / total_dataset["price"].std() < 1]
        #print ("4.", total_dataset.shape)

        # Impute missing values from here
        total_dataset = DataFrameImputer().fit_transform (total_dataset)

        # There are some columns with string values (E.g. Car type) -> need to label it as numerical labels
        total_dataset = MultiColumnLabelEncoder(columns = feature_need_label).fit_transform(total_dataset)

        # Standard scale dataset
        """scaled_data_set = total_dataset.copy()
        scaled_features = ['vehicle_mile', 'cylinder_disp','recovery_fee']
        scaled_data_set = scaled_data_set[scaled_features]

        scaler = StandardScaler()  
        scaler.fit(scaled_data_set.values)  
        total_dataset = scaler.transform(scaled_data_set.values)  """


        print ("Time for Loading dataset: %.3f" % (time.time() - stime))        

        self.total_dataset = total_dataset
    
    def get_total_dataset (self):
        
        return self.total_dataset
    
    """def get_training_dataset (self):
        
        self.traning_dataset = self.total_dataset.head (n = data_training_length)
        return self.traning_dataset
        # or return self.total_dataset[0:data_training_length]
    
    def get_validation_dataset (self):
        
        self.validation_dataset = self.total_dataset[data_training_length:data_training_length + data_validation_length]
        #print ('test:', self.validation_dataset)
        return self.validation_dataset
    
    def get_test_dataset (self):
        
        self.test_dataset = self.total_dataset.tail (data_test_length)
        return self.test_dataset""" 
    
    def encode_one_hot_feature (self, total_data_array, feature, feature_array):
        """
        There are some features need to be encoded (here using one-hot), because they are categorical features (represented as nominal, is not 
        meaningful because they are index, it doesn't make sense to add or subtract them ).
        NOTE: We need to know how many code of a features (based on training data set) -> fit correctly for test, validation dataset.
        
        feature_array: numpy array of feature 
        return the one-hot code for feature array
        """
        #print ("Encode one-hot.")
        #data_array = np.array ([self.get_total_dataset()[feature]]) # it is ok to use training dataset if the length of training dataset is large enough
        #data_array = self.impute_missing_values (feature, data_array, strategy_h)
        enc = OneHotEncoder(sparse = False)
        #print ("***", total_data_array.shape, feature_array.shape)
        enc.fit (total_data_array.T) 
        #print ('encoded:', enc.transform (feature_array.T))
        return enc.transform (feature_array.T)
        
        """X = np.array ([[128], [101], [105], [102], [109], [160], [101], [101], [102]])
        print ('X = ', X)
        print ('feature arr: ', feature_array)
        print (enc.fit_transform (feature_array.T))"""
    
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
        rating_code_arr = self.get_data_array (dataset, "rating_code")
        #price_2 = self.get_data_array (dataset, "price_2")
        #ident_car = np.concatenate ((model_code_arr, rating_code_arr, price_2), axis = 1 )
        #print ("ident_car", ident_car[:50])
        
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
                model_rating_dict[model_code_arr[i][0]] = [rating_code_arr[i][0]]
            elif rating_code_arr[i][0] not in model_rating_dict[model_code_arr[i][0]]:
                count += 1
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


    def encode_one_hot_car_ident (self, dataset):
        (count, car_ident_dict, car_ident_list) = self.count_car (dataset)
        #car_ident = self.create_car_ident (dataset)
        #car_ident_list = list (car_ident.reshape (car_ident.shape[0]))
        #print ("len of car_ident_list:", len (car_ident_list))
        print ("no different car identification:", len (Counter(car_ident_list).keys()))

        print ("count:", count, "car_ident_list:", car_ident_list[:100])
        #sys.exit (-1)
        enc = OneHotEncoder(sparse = False)
        return enc.fit_transform (np.array (car_ident_list).reshape (len (car_ident_list), 1)) 
        #return enc.fit_transform (car_ident) 

    def impute_missing_values (self, total_data_array, feature, feature_array, strategy):
        #TODO: More appropriate method
        """
        There are some columns with missing values ("NaN") -> need to impute those
            + by mean values
            + by median values
            + by most_frequent value
        
        feature_array: numpy array of feature 
        return the array of data without missing values
        """
        print ("Impute missing.")
        imp = Imputer(missing_values='NaN', strategy=strategy, axis=1)
        #data_array = np.array ([self.get_total_dataset()[feature]])
        imp.fit (total_data_array)
        """imp = DataFrameImputer ()
        imp.fit (pd.DataFrame (total_data_array))"""
        return imp.transform (feature_array) 

    def label_str_values (self, total_data_array, feature, feature_array):
        """
        There are some columns with string values (E.g. Car type) -> need to label it as numerical labels
        """
        #print ("Label str values.", feature)
        le = LabelEncoder ()
        #data_array = np.array ([self.get_total_dataset()[feature]])
        #data_array = self.impute_missing_values (feature, data_array, strategy_h)

        #print (total_data_array.shape, feature_array.shape)
        le.fit (total_data_array.reshape (total_data_array.shape[1], ))
        return le.transform (feature_array.reshape (feature_array.shape[1], )).reshape(1, feature_array.shape[1])

    def get_data_array (self, dataset, feature):  
        """
        dataset: training, validation, test, or total dataset
        feature: is a element in feature list
        return: an vector (1D numpy.array object)
        """
        #total_data_array = np.array ([self.get_total_dataset()[feature]])
        data_array = np.array ([dataset[feature]])
        total_data_array = data_array

        """if feature in feature_need_impute:
            #print ("@@")
            total_data_array = self.impute_missing_values (total_data_array, feature, total_data_array, strategy_h)
            if total_data_array.shape[0] != data_array.shape[0]:
                data_array = self.impute_missing_values (total_data_array, feature, data_array, strategy_h)
            else:
                data_array = total_data_array"""

        #print ("1.", data_array)
        """if feature in feature_need_label:
            total_data_array = self.label_str_values (total_data_array, feature, total_data_array)
            if total_data_array.shape[0] != data_array.shape[0]:
                data_array = self.label_str_values (total_data_array, feature, data_array)
            else:
                data_array = total_data_array
            #print ("2.", data_array)"""
        
        if using_one_hot_flag == 1:
            if feature in feature_need_encoding:
                if total_data_array.shape[0] != data_array.shape[0]:
                    print ("stop at function get_data_array()")
                    sys.exit (-1)
                    data_array = self.encode_one_hot_feature (total_data_array, feature, data_array) # TODO: Problem when use constraint
                else:
                    data_array = self.encode_one_hot_feature (data_array, feature, data_array)
                data_array = data_array.T

        
        #print ("no constraint:", data_array.T.shape, "feature:", feature)
        return data_array.T
    
    
    def get_data_array_with_constraint (self, dataset, feature, feature_constraint, feature_constraint_value): 
        """
        dataset: training, validation, test, or total dataset
        feature: is a element in feature list
        constraint: only get the input that satify the constraint
        feature_constraint_value: is only a value
        return: an vector (1D numpy.array object)
        """
        type_feature = type (feature_constraint_value)
        data_array_with_constraint = []# np.empty ([[]])
        data_array = self.get_data_array (dataset, feature)#np.array ([dataset[feature]]).T
        constraint_data_array = self.get_data_array (dataset, feature_constraint)#np.array ([dataset[feature_constraint]]).T
        for i in range (len (constraint_data_array)):
            if type_feature == str and constraint_data_array[i] == feature_constraint_value:
                #np.concatenate ((data_array_with_constraint, np.array ([[[data_array[i]])]]), axis = 0)
                data_array_with_constraint.append (data_array[i])
            elif type_feature == int and constraint_data_array[i] <= feature_constraint_value:
                data_array_with_constraint.append (data_array[i])
        #print ("constraint:", np.array (data_array_with_constraint).shape, "feature:", feature)
        return np.array (data_array_with_constraint)
    
    def get_data_array_remove_outliers (self, dataset, feature, window, all_outlier_index):
        """
        - Purpose: remove outliers by removing all data points with values of the corresponding feature which are out of the range from mean to +- window * SD (standard deviation)
        """
        data_array = self.get_data_array (dataset, feature)#np.array ([dataset[feature]]).T
        #print ("before remove outliers:", data_array.shape)
        len_data_array = data_array.shape[0]

        mean_data_array = np.mean (data_array)
        std_data_array = np.std (data_array)
        print ("mean:", mean_data_array, "std:", std_data_array)

        data_array_without_outliers = []
        outlier_index = []

        for i in range (len_data_array):
            if i not in all_outlier_index:
                if data_array[i] <= (mean_data_array + window * std_data_array) and data_array[i] >= (mean_data_array - window * std_data_array):
                    data_array_without_outliers.append (data_array[i])
                else:
                    #print ("---", data_array[i])
                    outlier_index.append (i)
        a = np.array (data_array_without_outliers)
        #print ("after remove outliers:", a.shape, "mean:", np.mean (a), "std:", np.std (a))
        #sys.exit (-1)

        
        return (np.array (data_array_without_outliers), outlier_index)
        
    
    def get_sale_duration_array (self, dataset):
        """
        - Purpose: as Return
        - Input: 
            + dataset: training, validation, test, or total dataset
        - Return: an vector oof sale duration as a numpy.array object
        """
        #print ("dataset", dataset)
        actual_advertising_date_array = self.get_data_array_with_constraint (dataset, "actual_advertising_date", "sale_state", "Sold-out")
        sale_date_array = self.get_data_array_with_constraint (dataset, "sale_date", "sale_state", "Sold-out")
        #print ("advertising date:", actual_advertising_date_array[:10])
        #print ("sale date:", sale_date_array[:10])
        #actual_advertising_date_array = self.get_data_array (dataset, "actual_advertising_date")
        #sale_date_array = self.get_data_array (dataset, "sale_date")
        length = len (sale_date_array)
        
        sale_duration_array = np.empty((length, 1))
        for i in range (length):
            sale_duration_array[i] = self.get_days_between (actual_advertising_date_array[i][0], sale_date_array[i][0])
        
        return sale_duration_array
        
    def get_data_array_with_constraints (self, dataset, feature, feature_constraint, feature_constraint_values): 
        """
        dataset: training, validation, test, or total dataset
        feature: is a element in feature list
        constraint: only get the input that satify the constraint
        feature_constraint_values: a list of accepted values
        return: an vector (1D numpy.array object)
        """
        
        data_array_with_constraint = []
        if feature  == "sale_duration":
            data_array = self.get_sale_duration_array (dataset)
        else:
            data_array = self.get_data_array (dataset, feature)
        
        constraint_data_array = self.get_data_array (dataset, feature_constraint)
        
        for i in range (len (constraint_data_array)):
            if is_in_range_constraint == 1:
                if constraint_data_array[i] in feature_constraint_values:
                    data_array_with_constraint.append (data_array[i])
                    
            elif is_in_range_constraint == 0:
                if constraint_data_array[i] not in feature_constraint_values:
                    data_array_with_constraint.append (data_array[i])
            
            else:
                pass
            
                
        return np.array (data_array_with_constraint)
    
        
    def get_data_matrix (self, dataset, features):
        """
        dataset: training, validation, test, or total dataset
        features: an array contains name of features 
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """
        return np.array (dataset[features])

    def get_data_matrix_car_ident (self, dataset):
        """
        dataset: training, validation, test, or total dataset
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """
        featureNo = len (features_remove_car_ident)
        
        """ NOTE!: The order of features are reversed due to concatenate() function => then we need to reverse it first"""
        features_copy = features_remove_car_ident[:]
        features_copy.reverse()

        car_ident_codes = np.array (dataset[car_ident]) 
        print ("car_ident_codes", car_ident_codes.shape)

        X1 = self.encode_one_hot_car_ident (dataset)
        d_ident = X1.shape[1]

        print ("X.shape1", X1.shape)
        X2 = np.array (dataset[features_remove_car_ident]) 
        X = np.concatenate ((X2, X1), axis = 1) 
        
        d_remain = X.shape[1] - d_ident
        print ("X.shape2", X.shape, d_remain, d_ident)
        return (car_ident_codes, X, d_ident, d_remain)

    def get_data_matrix_car_ident_flag (self, dataset):
        """
        dataset: the dataframe that we sorted by 'price_2' (concatenate train's price and test's predicted price from any method), and reconcatenate the testset with train dataset
        => return: Add 'set_flag' feature into the matrix.
        
        """
        print ("[get_data_matrix_car_ident_flag] dataset:", dataset[:2])
        car_ident_codes = np.array (dataset[car_ident])
        print ("car_ident_codes", car_ident_codes.shape)

        X_ident = self.encode_one_hot_car_ident (dataset)
        d_ident = X_ident.shape[1]

        X_ = np.array (dataset[feature_need_encoding])
        enc = OneHotEncoder(sparse = False)
        X_need_encoding = enc.fit_transform (X_) 

        X_need_not_encoding = np.array (dataset[['set_flag'] + ['price_2'] + ['price'] + features_not_need_encoding])
        """scaler = StandardScaler()  
        scaler.fit_transform(X_need_not_encoding)"""

        X = np.concatenate ((X_need_not_encoding, X_need_encoding, X_ident), axis=1)
        print ("[get_data_matrix_car_ident_flag] X:", X[:2])

        scaler = StandardScaler()  
        scaler.fit(X)  
        X = scaler.transform(X)

        print ("[Get data matrix car_ident_flag] X:", X[:2])
        sys.exit (-1)

        d_remain = X.shape[1] - d_ident - 3
        print ("[get_data_matrix_car_ident_flag] X.shape", X.shape, "d_remain", d_remain, "d_ident", d_ident)
        # The first column of X should be 'set_flag'
        return (car_ident_codes, X, d_ident, d_remain)

    def get_data_matrix_with_constraint (self, dataset, features, feature_constraint, feature_constraint_value): 
        """
        dataset: training, validation, test, or total dataset
        features: an array contains name of features 
        constraint: only get the input that satify the constraint
        feature_constraint_values: a list of accepted values
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """
        featureNo = len (features)
        
        """ NOTE!: The order of features are reversed due to concatenate() function => then we need to reverse it first"""
        features_copy = features[:]
        features_copy.reverse()
        
        X = self.get_data_array_with_constraint (dataset, features_copy[0], feature_constraint, feature_constraint_value)#np.array ([dataset[features[0]]]).T
        #print ('init X:', X)
        for i in range (1, featureNo):
            if features_copy[i] in self.headers:
                #X = np.concatenate ((np.array ([dataset[features[i]]]).T, X), axis = 1) # Input matrix (each column represents values of a feature, each row represents a data point)
                X = np.concatenate ((self.get_data_array_with_constraint (dataset, features_copy[i], feature_constraint, feature_constraint_value), X), axis = 1)
            else:
                break
        #print ('X:', X)
        return X
    
    def get_data_matrix_car_ident_with_constraint (self, dataset, feature_constraint, feature_constraint_value):
        """
        dataset: training, validation, test, or total dataset
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object), and the car identification is encoded as onehot
        """
        #TODO: add constraints 

        featureNo = len (features_remove_car_ident)
        
        """ NOTE!: The order of features are reversed due to concatenate() function => then we need to reverse it first"""
        features_copy = features_remove_car_ident[:]
        features_copy.reverse()
        
        X = self.encode_one_hot_car_ident (dataset.get_total_dataset ())
        d_ident = X.shape[1]
        print ("X.shape1", X.shape)
        for i in range (0, featureNo):
            if features_copy[i] in self.headers:
                a = self.get_data_array (dataset.get_total_dataset (), features_copy[i])
                #print ("a.shape", a.shape)
                X = np.concatenate ((self.get_data_array (dataset.get_total_dataset (), features_copy[i]), X), axis = 1)
            else:
                break
        print ("X.shape2", X.shape)
        return (X, d_ident)

    def get_data_matrix_without_outliers (self, dataset, features, window): 
        """
        dataset: training, validation, test, or total dataset
        features: an array contains name of features 
        constraint: only get the input that satify the constraint
        feature_constraint_values: a list of accepted values
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """
        featureNo = len (features)
        
        """ NOTE!: The order of features are reversed due to concatenate() function => then we need to reverse it first"""
        features_copy = features[:]
        features_copy.reverse()
        
        all_outlier_index = [] # initiate the empty list
        X, all_outlier_index = self.get_data_array_remove_outliers (dataset, features_copy[0], window, all_outlier_index) # update the outlier-index list with the first feature
        print ('X0:', X.shape)


        for i in range (1, featureNo):

            if features_copy[i] in self.headers: #try: #if features_copy[i] in self.headers:
                x, outlier_index = self.get_data_array_remove_outliers (dataset, features_copy[i], window, all_outlier_index) # update the outlier-index list with the all remaining features
                print ("...",x.shape, len (outlier_index))
        
                # update outlier index
                for index in outlier_index:
                    if index not in all_outlier_index:
                        all_outlier_index.append (index)

                X = np.concatenate ((x, X), axis = 1)
            else: #except ValueError: #else:
                #break
                print ("This feature is not exist!!!", features_copy[i])

            #except:
            #    print ("There are some errors here!!!")
        print ('Xn:', X.shape)
        return X
    
    def get_data_matrix_with_constraints (self, dataset, features, feature_constraint, feature_constraint_values): 
        """
        dataset: training, validation, test, or total dataset
        features: an array contains name of features 
        constraint: only get the input that satify the constraint
        feature_constraint_values: a list of accepted values
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """
        featureNo = len (features)
        
        """ NOTE!: The order of features are reversed due to concatenate() function => then we need to reverse it first"""
        features_copy = features[:]
        features_copy.reverse()
        
        X = self.get_data_array_with_constraints (dataset, features_copy[0], feature_constraint, feature_constraint_values)
        
        for i in range (1, featureNo):
            
            if features_copy[i] in self.headers:
                X = np.concatenate ((self.get_data_array_with_constraints (dataset, features_copy[i], feature_constraint, feature_constraint_values), X), axis = 1)
                
            else:
                break
        
        return X
    
    def get_expand_data (self, data):
        """
        Expand data by add ones-vector into the head of data
        """
        one = np.ones ((data.shape[0], 1))
        data_bar = np.concatenate ((one, data), axis = 1)
        return data_bar
    
    def __str__(self):
        """
        Display a string representation of the data.
        """
        output = 'Total dataset'
        return (output)


"""
    PROCESS DATA:
     - Implement, apply algorithms, build models
"""
class Support(Dataset):
    def __init__(self):
        
        """
            Initialize parameters for Training class
        """
        print ("Support class")
        Dataset.__init__(self)
    
    
    def get_predicted_label (self, tree, data):
        return  np.array (tree.predict(data)).reshape (data.shape[0], 1)

    def get_relative_err (self, predicted_label, actual_label):
        if actual_label.all() == 0:
            return -1
        return np.mean (np.abs (predicted_label - actual_label)/actual_label) * 100

    def get_smape (self, predicted_label, actual_label):
        """
            Symetric mean absolute percentage error: is an accuracy measure based on relative error (percentage error) to avoid the case that actual_label is 0
        """
        if (np.abs (actual_label) + np.abs (predicted_label)).all() == 0:
            return -1
        return np.mean (np.abs (predicted_label - actual_label) / (np.abs (actual_label) + np.abs (predicted_label)) ) * 100

    def get_err (self, err_type, predicted_label, actual_label):
        if err_type == "relative_err":
            return self.get_relative_err (predicted_label, actual_label)
        elif err_type == "smape":
            return self.get_smape (predicted_label, actual_label)

    def linear_regression (self, features, output)    :
        
        """
            Apply linear regression model 
        """
        X_train_data = self.get_data_matrix (self.traning_dataset, features)
        X_train_data_bar = self.get_expand_data (X_train_data)
        y = self.get_data_array (self.traning_dataset, output)
        
        # Apply built-in method in scikitlearn
        regr = linear_model.LinearRegression () #fit_intercept = False)
        regr.fit (X_train_data_bar, y)
        
        return regr
    
    def ridge_regression (self, features, output, l2_penalty = 1.5e-5)    :
        
        """
            Apply ridge regression model 
        """
        X_train_data = self.get_data_matrix (self.traning_dataset, features)
        X_train_data_bar = self.get_expand_data (X_train_data)
        y = self.get_data_array (self.traning_dataset, output)
        
        # Apply built-in method in scikitlearn
        regr = linear_model.Ridge (alpha = l2_penalty, normalize = True)
        regr.fit (X_train_data_bar, y)
        
        return regr
        
    def lasso_regression (self, features, output, l2_penalty = 1.5e-5)    :
        
        """
            Apply ridge regression model 
        """
        X_train_data = self.get_data_matrix (self.traning_dataset, features)
        X_train_data_bar = self.get_expand_data (X_train_data)
        y = self.get_data_array (self.traning_dataset, output)
        
        # Apply built-in method in scikitlearn
        regr = linear_model.Lasso (alpha = l2_penalty, normalize = True)
        regr.fit (X_train_data_bar, y)
        
        return regr
        
    def tree_GradientBoostingRegressor (self, train_data, train_label, test_data, test_label, n_estimators = 1000, learning_rate = 0.1, loss = 'lad'):
        
        """
            Apply GradientBoostingRegressor
        """        
        reg_tree = ensemble.GradientBoostingRegressor(n_estimators = n_estimators, learning_rate = learning_rate, loss = loss)
        stime = time.time()

        print ("training...")
        reg_tree.fit(train_data, train_label)
        print("Time for GradientBoostingRegressor learning_rate 0.1 tree fitting: %.3f" % (time.time() - stime))

        # Testing
        print ("testing...")
        err_type = "relative_err"
        print (train_data.shape, train_label.shape)
        predicted_train_label = self.get_predicted_label (reg_tree, train_data) 
        train_err = self.get_err (err_type, predicted_train_label, train_label)
        
        predicted_test_label = self.get_predicted_label (reg_tree, test_data)
        test_err = self.get_err (err_type, predicted_test_label, test_label) 
        print ("[DecisionTreeRegressor] Relative err: train_err: %.3f %%, test_err: %.3f %%" % (train_err, test_err))
        #sys.exit (-1)

        # assign set_flag and concatenate the dataset with both set_flag and price_2
        train_length = train_data.shape[0]
        test_length = test_data.shape[0]
        train_set_flag = np.zeros ((train_length, 1))
        test_set_flag = np.ones ((test_length, 1))
        train_set = np.concatenate ((train_set_flag, train_label, train_data, train_label), axis = 1)
        test_set = np.concatenate ((test_set_flag, predicted_test_label, test_data, test_label), axis = 1)
        
        # Concatenate to total_dataset
        total_dataset = np.concatenate ((train_set, test_set), axis = 0)
        total_length = total_dataset.shape[0]

        # return expanded dataframe
        columns = ['set_flag'] + ['price_2'] + features + ['price']
        index = [i for i in range (1, total_length + 1)]
        df = pd.DataFrame (data=total_dataset, index=index, columns=columns)
        return df
        #print (df)
        
        
    def __str__(self):
        """
            Display a string representation of the data.
        """
        output = ''
        return (output)

    
    
    
    
    
    
