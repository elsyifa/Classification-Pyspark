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

* Sample data
   If the dataset is too large, we can take sample of data. 
   Note: this step is optional.
* Check Missing Values.

* Handle Missing Values.

* Compare categorical variables in data train and data test.

* EDA 
  Create distribution visualization in each variables to get some insight of dataset.
  
* Handle insignificant categories in data train.

* Handle insignificant categories in data test.

* Handle outlier.

* Future Engineering.

* Split Data train to train and test.

* Modelling.

* Evaluation.

* Hyper-Parameter Tuning.

* Implementation Modelling to data test.
  

