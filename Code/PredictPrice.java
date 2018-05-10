import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;

import java.io.File;
import java.io.IOException;
import java.util.*;

import jxl.Cell;
import jxl.Sheet;
import jxl.Workbook;
import jxl.read.biff.BiffException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.IOException;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Range;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.ReplaceMissingWithUserConstant;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.Remove;

public class PredictPrice {
    private int len_rows, len_cols;
    private Instances trainInst, testInst, hash_inst;
    private double[] median = null;
    private double[] IQR = null;
    private HashMap<String, Integer> hm = null;
    private HashMap<Integer, String> hm2 = null;

    private int d_ident, d_remain;

    protected void computeIQR () {
        double[] values;
        double q1, q2, q3, iqr;
        int nuInst = this.trainInst.numInstances (); 
        int nuAttr = this.trainInst.numAttributes (); 
        System.out.println (nuInst + ", " + nuAttr);
        median = new double[nuInst];
        IQR = new double[nuInst];
        int[] sortedIndices;

        double x25 = 25.0/100 * (nuInst - 1) + 1;
        int modeX25 = (int) x25;
        double remainX25 = x25 - modeX25;

        double x50 = 50.0/100 * (nuInst - 1) + 1;
        int modeX50 = (int) x50;
        double remainX50 = x50 - modeX50;

        double x75 = 75.0/100 * (nuInst - 1) + 1;
        int modeX75 = (int) x75;
        double remainX75 = x75 - modeX75;

        //System.out.println (modeX25 + ", " + remainX25 + ", " + modeX50 + ", " + remainX50 + ", " + modeX75 + ", " + remainX75);

        for (int j = 0; j < nuAttr; j ++) {
            if (this.trainInst.attribute(j).type() == Attribute.NOMINAL) {
                this.median[j] = 0;
                this.IQR[j] = 0;
                continue;
            }

            // Sort attribute value
            values = this.trainInst.attributeToDoubleArray (j);
            sortedIndices = Utils.sort (values);

            // get q1, q2, q3 using 1st method to calculate interquartile
            q1 = values [sortedIndices[modeX25-1]] + remainX25 * (values [sortedIndices[modeX25]] - values [sortedIndices[modeX25-1]]);
            q2 = values [sortedIndices[modeX50-1]] + remainX50 * (values [sortedIndices[modeX50]] - values [sortedIndices[modeX50-1]]);
            q3 = values [sortedIndices[modeX75-1]] + remainX75 * (values [sortedIndices[modeX75]] - values [sortedIndices[modeX75-1]]);
            iqr = q3-q1;
            System.out.println (j + ": " + this.trainInst.attribute(j) + ": " + q1 + ", " + q2 + ", " + q3 + ", " + iqr);
            this.median[j] = q2;
            this.IQR[j] = iqr;
        }

    }

    public void printAttr (Instances inst, int numAttr) {
        for (int i = 0; i < numAttr; i ++) {
            System.out.println (inst.attribute (i));
        }
    }

    public void printInstances (String mess, Instances inst, int numInstance, int numAttr) {
        System.out.println (mess);
        for (int i = 0; i < numInstance; i ++) {
        //for (int i = 111036; i < 111047; i ++) {
            for (int j = 0; j < numAttr; j++) {
                System.out.printf (inst.instance(i).value(j) + ",");
            }
            System.out.printf ("\n");
        }
        System.out.printf ("\n\n");
}

    public void printArr (float[][] arr, int len_rows, int len_cols) {
        for (int i = 0; i < 1; i ++) {
            for (int j = 0; j < len_cols; j++) {
                System.out.printf (arr[i][j] + ",");
            }
            System.out.printf ("\n");
        }
    }

    
    public void printIqr (Instances inst, double[] median, double[] iqr, int numAttr) {
        for (int i = 0; i < numAttr; i ++) {
            if (inst.attribute (i).type() == Attribute.NOMINAL) {
                continue;
            }
            System.out.println (i + ", " + median [i] + ", " + iqr [i]);
        }
    }

    public static float[][] asArray (Instances inst) {
        int len_rows = inst.numInstances();
        int len_cols = inst.numAttributes();
        //System.out.println ("----Test:" + len_rows + "\t" + len_cols);
        float arr[][] = new float [len_rows][len_cols];
        for (int i = 0; i < len_rows; i++) {
            for (int j = 0; j < len_cols; j++) {
                arr[i][j] = (float) inst.instance(i).value(j); 
            }
        }
        return arr;
    }

    public void updateInstLen () throws Exception {
        this.len_rows = this.testInst.numInstances();
        this.len_cols = this.testInst.numAttributes();
    }

    public void convertNumToNom () throws Exception {
        String[] options = new String [2];
        NumericToNominal convert = new NumericToNominal();
        options [0] = "-R";
        // TODO: replace in the case of different test set structure. Eg. remove features
        options [1] = "1-15,49-53";
        convert.setOptions (options); // Can use: convert.setAttributeRange ("1-15,49-53");
        convert.setInputFormat (this.trainInst);
        this.trainInst = Filter.useFilter (this.trainInst, convert);
        this.testInst = Filter.useFilter (this.testInst, convert);
    }

    public void repMissValuesWithMeanMode () throws Exception {
        ReplaceMissingValues impute = new ReplaceMissingValues();
        impute.setInputFormat (this.trainInst);
        this.trainInst = Filter.useFilter (this.trainInst, impute);
        this.testInst = Filter.useFilter (this.testInst, impute);
        this.updateInstLen ();
    }

    public void repValues (int[][] newData) throws Exception {
        for (int i = 0; i < this.len_rows; i++) {
            for (int j = 0; j < this.len_cols; j++) {
                if (this.testInst.instance(i).isMissing (j)) {
                    this.testInst.instance(i).setValue(j, newData[i][j]);
                }
                // TODO: replace in the case of different test set structure. Eg. remove features
                if ((j >= 2 && j <= 5) || (j >= 7 && j <= 12)) {
                    //System.out.println ("---" + i + ", " + j + ": " + newData[i][j] + "--" + this.testInst.instance(i).value(j)); 
                    this.testInst.instance(i).setValue(j, newData[i][j]);
                }
            }
        }
        //this.printInstances ("here", this.testInst, 10, 53); 
        this.printInstances ("here", this.testInst, 10, this.len_cols); 
    }

    public void robustScale () throws Exception {
        double x;

        InterquartileRange iqr = new InterquartileRange ();
        iqr.setInputFormat (this.trainInst);
        Filter.useFilter (this.trainInst, iqr);

        // NOTE: weka was wrong when calculate q1, q3, then use our own functions
        for (int i = 0; i < this.len_rows; i++) {
            for (int j = 0; j < this.len_cols; j++) {
                // TODO: replace in the case of different test set structure. Eg. remove features
                if (j > 14 && j < 48) {
                    x = (this.testInst.instance(i).value(j) - this.median[j]);
                    if (this.IQR[j] != 0) {
                        x /= this.IQR[j];
                    }
                    this.testInst.instance(i).setValue(j, x);
                    //System.out.println (this.median[j] + ", " + this.IQR[j]);
                }
            }
        }
        this.updateInstLen ();
    }

    public void oneHotEncode () throws Exception {
        String[] options = new String [1];
        NominalToBinary encode = new NominalToBinary ();
        options [0] = "-A";
        encode.setOptions (options);
        encode.setInputFormat (this.trainInst);

        this.trainInst = Filter.useFilter (this.trainInst, encode);
        this.testInst = Filter.useFilter (this.testInst, encode);
        this.updateInstLen ();
    }

    public double getRmse (float[] prediction, float[] actual, int len) {
        double diff, rmse = 0;
        for (int i = 0; i < len; i ++) {
            diff = prediction[i] - actual[i];
            rmse += diff * diff;
        }
        rmse = Math.sqrt (rmse/len);
        return rmse;
    }

    public double getMae (float[] prediction, float[] actual, int len) {
        double diff, mae = 0;
        for (int i = 0; i < len; i ++) {
            diff = prediction[i] - actual[i];
            mae += Math.abs (diff);
        }
        mae = mae/len;
        return mae;
    }

    public double getRelErr (float[] prediction, float[] actual, int len) {
        // NOTE: need to corespond to the function used in python
        double diff, relErr = 0;
        for (int i = 0; i < len; i ++) {
            diff = prediction[i] - actual[i];
            relErr += Math.abs (diff) / Math.max (actual[i], prediction[i]) * 100;
        }
        relErr = relErr/len;
        return relErr;
    }

    public double getSmape (float[] prediction, float[] actual, int len) {
        double diff, smape = 0;
        for (int i = 0; i < len; i ++) {
            diff = prediction[i] - actual[i];
            smape += Math.abs (diff) / (actual[i] + prediction[i]) * 100;
        }
        smape = smape/len;
        return smape;
    }

    public double[] getErr (float[] prediction, float[] actual, int len) {
        double[] err = new double[4];
        err[0] = this.getRmse (prediction, actual, len);
        err[1] = this.getMae (prediction, actual, len);
        err[2] = this.getRelErr (prediction, actual, len);
        err[3] = this.getSmape (prediction, actual, len);
        return err;
    }

    public String[][] RepMisValues (String hashReplaceFile,String[][] data, int len_rows, int len_cols) {
        String[][] newData = new String [len_rows][len_cols];
        this.hm2 = new HashMap<>();
        CSVReader hashCode = new CSVReader ();

        // The hash_code file and the input file must be encoded as UTF-8 (in python, use to_csv() function with encoding parameter to specify) (NOTE: don't modify these 2 files manually, otherwise the format of the file will be broken)
        hashCode.setInputFile (hashReplaceFile);
        this.hm2 = hashCode.hashMap_2 (1, 53); // Work
        //this.hm2 = hashCode.hashMap_2 (1, len_cols); // If there are no label column at the end
        //this.hm2 = hashCode.hashMap_2 (1, len_cols-1); // If there are the label column at the end

        for (int i = 0; i < len_rows; i++) {
            for (int j = 0; j < len_cols; j++) {

                //System.out.println (i + ", " + j + ": " + data[i][j]);
                // Replace missing values
                if (data[i][j] == null || data[i][j].equals ("")) {
                    try {
                        newData[i][j] = this.hm2.get (j);
                    } catch (Exception e) {
                        e.printStackTrace ();
                    }
                    System.out.println (i + ", " + j + ": " + data[i][j] + "- " + newData[i][j]);
                }
                else {
                    newData[i][j] = data[i][j];
                }
            }
        }
        return newData;
    }

    public int[][] RepStrToCatValues (String hashStringFile, String[][] data, int len_rows, int len_cols) {
        int[][] newData = new int [len_rows][len_cols];
        this.hm = new HashMap<>();
        CSVReader hashCode = new CSVReader ();

        // The hash_code file and the input file must be encoded as UTF-8 (in python, use to_csv() function with encoding parameter to specify) (NOTE: don't modify these 2 files manually, otherwise the format of the file will be broken)
        hashCode.setInputFile (hashStringFile);
        // TODO: if the train set is changed, the hash code may change, need to update this file, and the number of hash codes (Eg. 185)
        this.hm = hashCode.hashMap (185, 54); // Work
        //this.hm = hashCode.hashMap (185, len_cols);

        for (int i = 0; i < len_rows; i++) {
            for (int j = 0; j < len_cols; j++) {

                //System.out.println (i + ", " + j + ": " + data[i][j]);
                // TODO: replace in the case of different test set structure. Eg. remove features
                if ((j >= 2 && j <= 5) || (j >= 7 && j <= 12)) {
                    try {
                        if (this.hm.containsKey (data[i][j])) {
                            newData[i][j] = this.hm.get (data[i][j]);
                        }
                        else {
                            System.out.println (i + "-" + j + data[i][j] + ": not found!!!");
                            System.exit (-1);
                        }
                    } catch (Exception e) {
                        e.printStackTrace ();
                    }
                    // Just for testing
                    if (i >= 111038 && i <= 111047 && j == 2) {
                        System.out.println (i + ", " + j + ": " + data[i][j] + "- " + newData[i][j]);
                    }
                }
                
                else {
                    if (data[i][j] == null || data[i][j].equals ("")) {
                        newData[i][j] = 0;
                    }
                    else {
                        newData[i][j] = (int) Float.parseFloat (data[i][j]);
                    }
                }
            }
        }
        return newData;
    }

    //public void prepTest (String trainFile, String testFile, String hashStringFile, String hashReplaceFile, int lenTest, int lenFeatures, int valdFlag) {
    public float[] prepTest (String trainFile, String testFile, String hashStringFile, String hashReplaceFile, int lenTest, int lenFeatures, int valdFlag)  throws Exception {
        try {

            // Read test data into an array of string
            CSVReader file = new CSVReader ();
            file.setInputFile (testFile);
            file.read (lenTest, lenFeatures, 2);
            String[][] origData  = file.getStrArr ();

            String[][] repMisData = new String [lenTest][lenFeatures];
            repMisData = this.RepMisValues (hashReplaceFile, origData, lenTest, lenFeatures);
            
            // Using hash function to transform Korean to numeric values, because weka will consider string as number (top-down rule) -> not match with train data (using python) that sorted acording to Alphabet
            int[][] repStrToCatData = new int [lenTest][lenFeatures];
            repStrToCatData = this.RepStrToCatValues (hashStringFile, repMisData, lenTest, lenFeatures);


            // Use weka APIs to load the dataset into the Instances
            CSVLoader loader = new CSVLoader ();

            // Load the train set into the Instances 
            loader.setSource (new File (trainFile)); 
            this.trainInst = loader.getDataSet ();
            
            // Load the test set into the Instances 
            loader.setSource (new File (testFile));
            this.testInst = loader.getDataSet ();

            // Update the size of the testset
            this.updateInstLen ();

            // TODO: if the last column is label, the len of feature will be lenFeatures
            //this.printInstances ("Read", this.testInst, 10, lenFeatures-1); // have label: will work
            //this.printInstances ("Read", this.testInst, 10, lenFeatures);
            //this.printInstances ("Read", this.testInst, 10, 53); // Only print 10 first rows 
            this.printInstances ("Read", this.testInst, 10, this.len_cols); // After updating the lsize of the dataset, this line will work
            this.printAttr (this.testInst, this.len_cols); 

            // Replace Korean of the test instance to numeric values
            this.repValues (repStrToCatData);

        } catch (Exception e) {
            e.printStackTrace();
            //System.out.printf ("Error: Can not read input file!\n1. CHeck the name of input file [total or train].\n 2. Check header names, it should be different.\n 3. Check the consistent format between train and test files.\n\n");
        }
        
        this.printInstances ("Rep", this.testInst, 10, this.len_cols); 


        // If it is used to validate (have label, and calculate error, then valdFlag==1
        double[] testLabel = new double[this.len_rows];
        float[] testLabelArr = new float[this.len_rows];
        if (valdFlag == 1) {
            // Seperate the dataset into data and label
            testLabel = this.testInst.attributeToDoubleArray (this.len_cols-1);
            for (int i = 0; i < this.len_rows; i ++) {
                testLabelArr [i] = (float) testLabel [i];
            }
    
            // Try with Remove label if the last column is label
            this.trainInst.deleteAttributeAt (this.len_cols-1); 
            this.testInst.deleteAttributeAt (this.len_cols-1);
        }

        // Update the size of the testset
        this.updateInstLen ();

        // set types of attributes
        this.convertNumToNom ();
        //this.printInstances ("After Convert num-nom: ", this.testInst, this.len_rows, this.len_cols); 
       
        // Compute IQR, median
        this.computeIQR ();

        // Robust scale: numeric attributes
        this.robustScale ();
        this.printInstances ("After scaling: ", this.testInst, 10, this.len_cols); 
        
        // One hot encode the nominal attributes
        this.oneHotEncode ();

        System.out.println ("Test: " + this.len_rows + ", " + this.len_cols);
        return testLabelArr;
    }

    public float[] predictTest (String modelFile, float[] testLabelArr, int lenIdent, int lenRemain) {
        //========================================================
        //==== Load data into the pre-trained model
        //========================================================
        this.d_ident = lenIdent;
        this.d_remain = lenRemain;
        System.out.println ("Check: " + this.len_rows + ", " + this.len_cols);
        float[][] test_arr = new float [this.len_rows][this.len_cols];
        float[][] test_ident_arr = new float [this.len_rows][this.d_ident];
        float[][] test_remain_arr = new float [this.len_rows][this.d_remain];

        test_arr = this.asArray (this.testInst);
        
        // Seperate data into ident and remain parts
        for (int i = 0; i < this.len_rows; i++) {
            for (int j = 0; j < this.len_cols; j++) {
                if (j < this.d_remain) {
                    test_remain_arr[i][j] = test_arr[i][j];
                }
                else {
                    test_ident_arr[i][j-this.d_remain] = test_arr[i][j];
                }
            }
        }


        // Model must be saved at the same folder with the source file
        float[][] pred_y = new float[this.len_rows][1];
        float[] testPredLabelArr = new float[this.len_rows];
        try (SavedModelBundle b = SavedModelBundle.load (modelFile, "serve"); Session s = b.session()) {
            
            try (Tensor tx = Tensor.create(test_ident_arr);
                Tensor ty = Tensor.create(test_remain_arr);
                Tensor tz = Tensor.create(testLabelArr);
                Tensor to = s.runner().feed("x_ident", tx).feed("x_remain", ty).feed("Y", tz).fetch("prediction").run().get(0)) {
                to.copyTo (pred_y);
                
                System.out.printf ("Actual\t Prediction\n");
                for (int i = 0; i < this.len_rows; i ++) {
                    // Only print the first 10 results
                    if (i < 10) {
                        System.out.printf (testLabelArr[i] + "\t" + pred_y[i][0] + "\n");
                    }
                    testPredLabelArr[i] = pred_y[i][0];
                }
            }
        }
        double[] err = new double[4];
        err = this.getErr (testPredLabelArr, testLabelArr, this.len_rows);
        System.out.println ("\n\nrmse: " + err[0]);
        System.out.println ("mae: " + err[1]);
        System.out.println ("rel_err: " + err[2]);
        System.out.println ("smape: " + err[3]);//*/

        return testPredLabelArr;
    } 

    public static void main (String[] args) throws Exception {

        PredictPrice obj1 = new PredictPrice();
        System.out.println("Using TensorFlow, Weka");
        String trainFile = "./total.csv_beforeImpute"; 
        String testFile = "./test.csv";
        String hashStringFile = "./hash_code1.csv"; 
        String hashReplaceFile = "./hash_code2.csv";
        int lenTest = 27759;
        int lenFeatures = 54; 
        int valdFlag = 1;
        float[] testLabelArr = new float[lenTest];

        // Load and preprocess the test data
        testLabelArr = obj1.prepTest (trainFile, testFile, hashStringFile, hashReplaceFile, lenTest, lenFeatures, valdFlag);

        
        String modelFile = "./";
        int lenIdent = 1920, lenRemain = 595;
        float[] testPredLabelArr = new float[obj1.len_rows];

        // Load the pre-trained model and give the prediction based on data of the test file
        testPredLabelArr = obj1.predictTest (modelFile, testLabelArr, lenIdent, lenRemain);
    }
}
            

class CSVReader {
    private String inputFile;
    private String[] str = null;
    private String[][] strArr = null;
    private float[][] data = null;
    private float[] label = null;

    public void setInputFile (String inputFile) {
        this.inputFile = inputFile;
    }

    public String[] getStr () {
        return this.str;
    }

    public String[][] getStrArr () {
        return this.strArr;
    }

    public float[][] getData () {
        return this.data;
    }

    public float[] getLabel () {
        return this.label;
    }

    public HashMap<String, Integer> hashMap (int noRowsHash, int noCols) {
        HashMap<String, Integer> map = new HashMap<>();
        this.strArr = new String[noRowsHash][noCols];

        this.read (noRowsHash, noCols, 2);

        for (int i = 0; i < noRowsHash; i++) {
            for (int j = 0; j < noCols; j++) {
                if ((j >= 2 && j <= 5) || (j >= 7 && j <= 12)) {
                    if (this.strArr[i][j] == null || this.strArr[i][j].equals ("")) {
                        //System.out.println (i + ", " + j + ": " + this.strArr[i][j]);
                        continue;
                    }
                    map.put (this.strArr[i][j], i);
                }
            }
        }
        return map;
    }

    public HashMap<Integer, String> hashMap_2 (int noRowsHash, int noCols) {
        HashMap<Integer, String> map = new HashMap<>();
        this.strArr = new String[noRowsHash][noCols];

        this.read (noRowsHash, noCols, 2);

        for (int i = 0; i < noRowsHash; i++) {
            for (int j = 0; j < noCols; j++) {
                if (this.strArr[i][j] == null || this.strArr[i][j].equals ("")) {
                    //System.out.println (i + ", " + j + ": " + this.strArr[i][j]);
                    continue;
                }
                map.put (j, this.strArr[i][j]);
            }
        }
        return map;
    }

    public void read (int noRows, int noCols, int flag) {
        this.data = new float[noRows][noCols];
        this.label = new float[noRows];
        this.str = new String[noCols];
        this.strArr = new String[noRows][noCols];
        String line = "";
        String cvsSplitBy = ",";
        int i = 0, j = 0, lenString;

        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(this.inputFile), "UTF-8"))) {
            while ((line = br.readLine()) != null) {
                // use comma as separator
                String[] s = line.split (cvsSplitBy);
                lenString = s.length;
                for (j = 0; j < lenString; j ++) {
                    if (flag == 1) {
                        if (j < lenString - 1) {
                            this.data[i][j] = Float.parseFloat (s[j]);
                        }
                        else {
                            this.label[i] = Float.parseFloat (s[j]);
                        }
                    }
                    else if (flag == 0 && i > 0) {
                        this.str[j] = s[j];
                    }
                    else if (flag == 2 && i > 0) {
                        this.strArr[i-1][j] = s[j];
                    }
                }
                i ++;
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}

