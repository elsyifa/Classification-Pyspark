# Classification-Pyspark
This repository of classification template using pyspark.

In this part, I tried to make a template of classification in machine learning using pyspark. I will try to explain step by step from load data, data cleansing and making prediction. I created some functions in pyspark to make an automatization, so users just need to update or replace the dataset.

To test my template, I used data Home_Quote_Conversion from Kaggle https://www.kaggle.com/c/homesite-quote-conversion. This dataset represent the activity who are interested in buying policies from Homesite. QuoteConversion_Flag indicates whether the customer purchased a policy and the task is to predict QuoteConversion_Flag for each QuoteNumber in the test set.

In general, the steps of classification in machine learning are:

* Load Data into Spark Dataframe.
  Because we will work on spark environment so the dataset must be in spark dataframe. In this step, I created function to load data into spark dataframe. To run this function, first we have to define type of file of dataset (text or parquet) and path where dataset is stored and delimeter ',' or other. 
  
* Check the data.
  After load data, lets do some check of the dataset such as numbers of columns, numbers of observations, names of columns, type of columns, etc.
  
* Define categorical and numerical variables.
  In this step, I tried to split the variables based on it's data types. If data types of variables is string will be saved in list called **cat_cols** and if data types of variables is integer or double will be saved in list called **num_cols**. This split applied on data train and data test. This step applied to make easier in the following step so I don't need to define categorical and numerical variables manually.

* Sample data
   If the dataset is too large, we can take sample of data. 
   Note: this step is optional.
   
* Check Missing Values.
  Sometimes the data received is not clean. So, we need to check whether missing values or not. Output from this step is the name of columns which have missing values and the number of missing values. To check missing values, actually I created two method:
   a. Using pandas dataframe, 
   b. Using pyspark dataframe.
  But the prefer method is method using pyspark dataframe so if dataset is too large we can still calculate / check missing values.

* Handle Missing Values.
  The approach that used to handle missing values between numerical and categorical variables is different. For numerical variables I fill the missing values with average in it's columns. While for categorical values I fill missing values use most frequent category in that column, therefore count categories which has max values in each columns is needed.
  
* Compare categorical variables in data train and data test.
  In this step, we check whether categories between data train and data test same or not. Categories in data test will be equated with data train. This step is needed to avoid error in feature engineering, if there are differences categories between data train and data test the error will appear at feature engineering process in data test so the modelling process cannot be applied in data test.
  
* EDA 
  Create distribution visualization in each variables to get some insight of dataset.
  
* Handle insignificant categories in data train.
  Sometimes there are categories with very few amount, those categories is called insignificant categories. Those insignificant categories will be replaced with the largest numbers of catories in each categorical columns. Sometimes this replacing will make better modelling. 
  Note: the determination of threshold that category have very few amount is based on trial n error. In this case I used threshold 98% for maximum amount and 0.7% for minimum amount. Each categories in a column that have percentage under 0.7% will be replaced with category that has percentage equal or lower than 98%.
  
* Handle insignificant categories in data test.
  To handle insignificant categories in data test, I refer to insignificant categories in data train. Categories that replaced will be equated with data train to avoid differences categories between data train and data test. As known those differences will trigger error in feature angineering and modelling process.
  
* Handle outlier.
  Outlier is observations that fall below lower side or above upper side.
  To handle outlier we approach by replacing the value greater than upper side with upper side value and also replacing the value lower than lower side with lower side value. So, we need calculate upper and lower side from quantile value, quantile is probability distribution of variable. In General, there are three quantile:

  Q1 = the value that cut off 25% of the first data when it is sorted in ascending order.
  Q2 = cut off data, or median, it's 50 % of the data
  Q3 = the value that cut off 75% of the first data when it is sorted in ascending order.
  IQR or interquartile range is range between Q1 and Q3. IQR = Q3 - Q1.

  Upper side = Q3 + 1.5 * IQR
  Lower side = Q1 - 1.5 * IQR

  To calculate quantile in pyspark dataframe I created a function and then created function to calculate uper side, lower side, replacing upper side and replacing lower side. function of replacing upper side and lower side will looping as much as numbers of numerical variables in dataset (data train or data test).

* Feature Engineering.
  Before splitting the data train, all categorical variables must be made numerical. There are several approaches to categorical variables in SparkML, including:
  - StringIndexer, which is to encode the string label into the index label by sequencing the string frequency descending and giving the smallest index (0) at most string frequency.
  - One-hot Encoding, which is mapping the label column (string label) on the binary column.
  - Vector assembler, which is mapping all columns in vector.


* Split Data train to train and test.

* Modelling.

* Evaluation.

* Hyper-Parameter Tuning.

* Implementation Modelling to data test.
  

