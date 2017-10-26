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
from Neural_network import *
"""
    LOAD DATA:
     - Load data from csv file, and store as narrays
"""

class Data_preprocessing (object):
    def __init__(self):
        pass

    def get_days_between(self, d1, d2):
        
        if np.isnan(d1) or np.isnan(d2):
            return (-1 )
        str_d1 = str (int (d1))
        str_d2 = str (int (d2))
        d1 = datetime.strptime(str_d1, "%Y%m%d")
        d2 = datetime.strptime(str_d2, "%Y%m%d")
        return int (abs((d2 - d1).days))
        

class Dataset (Data_preprocessing):
    def __init__(self):
        """
        Loading data from csv file
        skip_rows: the number of row will be skipped from row 0
        skip_footer: the number of row will be skipped from the end
        """
        stime = time.time()
        self.headers = full_features
        dtype_dict = full_features_dict
        #total_dataset = pd.read_excel (dataset_excel_file, names = self.headers, dtype = dtype_dict, header = 0)
        total_dataset = pd.read_excel (dataset_excel_file, names = self.headers, converters = dtype_dict, header = 0)
        print ("Time for Loading dataset: %.3f" % (time.time() - stime))        
        self.total_dataset = total_dataset
        self.traning_dataset = self.get_training_dataset()
        self.validation_dataset = self.get_validation_dataset()
        self.test_dataset = self.get_test_dataset()
    
    def get_total_dataset (self):
        
        return self.total_dataset
    
    def get_training_dataset (self):
        
        self.traning_dataset = self.total_dataset.head (n = data_training_length)
        return self.traning_dataset
        # or return self.total_dataset[0:data_training_length]
    
    def get_validation_dataset (self):
        
        self.validation_dataset = self.total_dataset[data_training_length:data_training_length + data_validation_length]
        #print ('test:', self.validation_dataset)
        return self.validation_dataset
    
    def get_test_dataset (self):
        
        self.test_dataset = self.total_dataset.tail (data_test_length)
        return self.test_dataset 
    
    def encode_one_hot_feature (self, feature, feature_array):
        """
        There are some features need to be encoded (here using one-hot), because they are categorical features (represented as nominal, is not 
        meaningful because they are index, it doesn't make sense to add or subtract them ).
        NOTE: We need to know how many code of a features (based on training data set) -> fit correctly for test, validation dataset.
        
        feature_array: numpy array of feature 
        return the one-hot code for feature array
        """
        training_data_array = np.array ([self.get_training_dataset()[feature]])
        #print ('training_data_array', training_data_array.T)
        #print ('feature array:', feature_array.T)
        enc = OneHotEncoder()
        enc.fit (training_data_array.T)
        #print ('encoded:', enc.transform (feature_array.T).toarray())
        return enc.transform (feature_array.T).toarray()
        
        """X = np.array ([[128], [101], [105], [102], [109], [160], [101], [101], [102]])
        print ('X = ', X)
        print ('feature arr: ', feature_array)
        print (enc.fit_transform (feature_array.T))"""
        
    def get_data_array (self, dataset, feature):  
        """
        dataset: training, validation, test, or total dataset
        feature: is a element in feature list
        return: an vector (1D numpy.array object)
        """
        data_array = np.array ([dataset[feature]])
        if using_one_hot_flag == 1:
            if feature in feature_need_encoding:
                return self.encode_one_hot_feature (feature, data_array)
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
        return np.array (data_array_with_constraint)
    
    def get_sale_duration_array (self, dataset):
        """
        - Purpose: as Return
        - Input: 
            + dataset: training, validation, test, or total dataset
        - Return: an vector oof sale duration as a numpy.array object
        """
        """actual_advertising_date_array = self.get_data_array_with_constraint (dataset, "actual_advertising_date", "sale_state", "Sold-out")
        sale_date_array = self.get_data_array_with_constraint (dataset, "sale_date", "sale_state", "Sold-out")"""
        actual_advertising_date_array = self.get_data_array (dataset, "actual_advertising_date")
        sale_date_array = self.get_data_array (dataset, "sale_date")
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
    
    '''def get_data_array_without_onehot_with_constraint_array (self, dataset, feature, feature_constraint_array, feature_constraint_value_array): 
        """
        dataset: training, validation, test, or total dataset
        feature: is a element in feature list
        constraint: only get the input that satify the constraint
        feature_constraint: an array of features on the constraint
        feature_constraint_value: an array of corresponding constraint values with feature in feature_constraint
        return: an vector (1D numpy.array object)
        """
        if len (feature_constraint_array) != len (feature_constraint_value_array):
            raise ValueError('Length of feature_constraint_array should equal to length of feature_constraint_value_array")
        data_array_with_constraint = []# np.empty ([[]])
        data_array = self.get_data_array_without_onehot (dataset, feature)#np.array ([dataset[feature]]).T
        len_feature_constraint_array = len (feature_constraint_array)
        
        for j in range (len_feature_constraint_array): 
            constraint_data_array = self.get_data_array_without_onehot (dataset, feature_constraint_array[j])#np.array ([dataset[feature_constraint]]).T
            type_feature = type (feature_constraint_value)
            
            
            len_constraint_data_array = len (constraint_data_array)
            for i in range (len_constraint_data_array):
                if type_feature == str and constraint_data_array[i] == feature_constraint_value:
                    data_array_with_constraint.append (data_array[i])
                elif type_feature == int and constraint_data_array[i] <= feature_constraint_value:
                    data_array_with_constraint.append (data_array[i])
        return np.array (data_array_with_constraint)'''
    
    def get_data_matrix (self, dataset, features):
        """
        dataset: training, validation, test, or total dataset
        features: an array contains name of features 
        => return: a matrix with rows are data points, columns are features values (nD numpy.array object)
        
        """
        featureNo = len (features)
        
        """ NOTE!: The order of features are reversed due to concatenate() function => then we need to reverse it first"""
        features_copy = features[:]
        features_copy.reverse()
        
        X = self.get_data_array (dataset, features_copy[0])#np.array ([dataset[features[0]]]).T
        #print ('init X:', X)
        for i in range (1, featureNo):
            if features_copy[i] in self.headers:
                #X = np.concatenate ((np.array ([dataset[features[i]]]).T, X), axis = 1) # Input matrix (each column represents values of a feature, each row represents a data point)
                X = np.concatenate ((self.get_data_array (dataset, features_copy[i]), X), axis = 1)
            else:
                break
        #print ('X:', X)
        return X
    
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
class Training(Dataset):
    def __init__(self):
        
        """
            Initialize parameters for Training class
        """
        Dataset.__init__(self)
    
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
        
    def tree_GradientBoostingRegressor (self, features, output, n_estimators = 500, learning_rate = 0.1, loss = 'ls'):
        
        """
            Apply GradientBoostingRegressor
        """        
        X_train_data = self.get_data_matrix (self.traning_dataset, features)
        X_train_data_bar = self.get_expand_data (X_train_data)
        y = self.get_data_array (self.traning_dataset, output)
                
        regr = ensemble.GradientBoostingRegressor(n_estimators = n_estimators, learning_rate = learning_rate, loss = loss)
        regr.fit(X_train_data_bar,y)
        
        return regr
    
    """def neural_network_regression (self, hidden_layer_sizes, X, y):
        nn = build_model(3, print_loss=True)
        return nn"""
        
    def __str__(self):
        """
            Display a string representation of the data.
        """
        output = ''
        return (output)

    
    
    
    
    
    
