JC = javac
JCR = java
JNI_DIR = ./jni # Java Native Interface folder 

TRAIN_FILE = ./total.csv_beforeImpute_33features # training file
TEST_FILE = ./test.csv_33                        # test file, the structure should be same as the training file

HASH_STRING_FILE = ./hash_string_33.csv # the file used to map string values to categorical values (eg. car type, fuel type...)
HASH_SPACE_FILE = ./hash_replace_33.csv # the file used to map missing values to the average or the median of the corresponding feature in the training file

NU_TEST_ITEMS = 22525 # #rows of the test file
NU_COL_TEST_FILE = 36 # #columns of the test file

VALID_FLAG = 1 # set to 0 if only need to give the prediction

D_IDENT = 5638 # the dimension of the identification vector (maker, class, car, model, grade codes) after onehot encoding. Need to change if change train file
D_REMAIN = 512 # the dimension of all the other features

# Prepare
# TODO: change Compare file to run it easier, or run this file in the train/test step using python (with a_Run.sh file)
#prepare:
#	python Compare.py

# Compile
#all: prepare
all:
	$(JC) -cp ".:*" PredictPrice.java

# Modify parameters and execute
exec: 
	$(JCR) -cp ".:*" -Djava.library.path=$(JNI_DIR) PredictPrice $(TRAIN_FILE) $(TEST_FILE) $(HASH_STRING_FILE) $(HASH_SPACE_FILE) $(NU_TEST_ITEMS) $(NU_COL_TEST_FILE) $(VALID_FLAG) $(D_IDENT) $(D_REMAIN) 

# Pass the arguments into the Makefile
exec_arg: 
	$(JCR) -cp ".:*" -Djava.library.path=$(JNI_DIR) PredictPrice $(TRAIN_FILE_ARG) $(TEST_FILE_ARG) \
        $(HASH_STRING_FILE_ARG) $(HASH_SPACE_FILE_ARG) $(NU_TEST_ITEMS_ARG) $(NU_COL_TEST_FILE_ARG) $(VALID_FLAG_ARG) $(D_IDENT_ARG) $(D_REMAIN_ARG) 

# Eg. make exec_arg TRAIN_FILE_ARG=./total.csv_beforeImpute_33features TEST_FILE_ARG=./test.csv_33 HASH_STRING_FILE_ARG=./hash_string_33.csv HASH_SPACE_FILE_ARG=./hash_replace_33.csv NU_TEST_ITEMS_ARG=22525 NU_COL_TEST_FILE_ARG=36 VALID_FLAG_ARG=1 D_IDENT_ARG=5638 D_REMAIN_ARG=512

clean:
	$(RM) *.class


.PHONY: all clean exec
