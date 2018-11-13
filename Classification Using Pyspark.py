#Classification Using Pyspark

#Pyspark Initializasing
# to make pyspark importable as a regular library
import findspark
findspark.init()

import pyspark

from pyspark import SparkContext
sc = SparkContext.getOrCreate()

#initializasing SparkSession for creating Spark DataFrame
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


#Load Libraries
# Data Frame spark profiling 
from pyspark.sql.types import IntegerType, StringType, DoubleType, ShortType, DecimalType
import pyspark.sql.functions as func
from pyspark.sql.functions import isnull
from pyspark.sql.functions import isnan, when, count, col, round
from pyspark.sql.functions import mean
from pyspark.sql.types import Row
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf

# Pandas DF operation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array

# Modeling + Evaluation
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.sql.functions import when
from pyspark.sql import functions as F
from pyspark.sql.functions import avg
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from pyspark.sql import Window
from pyspark.sql.functions import rank,sum,col
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorSlicer

window = Window.rowsBetween(Window.unboundedPreceding,Window.unboundedFollowing)


#Load Data to Spark DataFrame
#Initializing File Type and path for data train
file_type = 'text'
path=r'train.csv'
delimeter=','

def load_data(file_type):
    """input type of file "text" or "parquet" and Return pyspark dataframe"""
    if file_type =="text": # use text as file type input
        df = spark.read.option("header", "true") \
                       .option("delimeter",delimeter)\
                       .option("inferSchema", "true") \
                       .csv(path)  #path file that you want import
    else:  
        df= spark.read.parquet("example.parquet") #path file that you want import
    return df

#call function load_data
df = load_data(file_type)

#Initializing File Type and path for data test
file_type = 'text'
path=r'test.csv'
delimeter=','

#call function load_data
test_data = load_data(file_type)


#Check data
#check type of data train and data test
type(df)
type(test_data)

#show 5 observation in data train
df.show(5)

#show 5 observation in data test
test_data.show(5)

#Print Schema and count number of columns from data train
len(df.columns), df.printSchema()

#Print Schema and count number of columns from data test
len(test_data.columns), test_data.printSchema()


#drop column Original_Quote_Date from data train
df_final=df.drop('Original_Quote_Date')

#count number of observation in data train
df_final.count()

#drop column Original_Quote_Date from data test
test_data=test_data.drop('Original_Quote_Date')

#calculate percentage of target and save in dataframe called target_percent
target_percent=df_final.groupBy('label').count().sort(col("count").desc())\
                        .withColumn('total',sum(col('count')).over(window))\
                        .withColumn('Percent',col('count')*100/col('total')) 

#show dataframe terget_percent to check the proportion
target_percent.show()


#Define categorical and nummerical variable in df_final (data train)
#Categorical and numerical variable
#just will select string data type
cat_cols = [item[0] for item in df_final.dtypes if item[1].startswith('string')] 
print("cat_cols:", cat_cols)

#just will select integer or double data type
num_cols = [item[0] for item in df_final.dtypes if item[1].startswith('int') | item[1].startswith('double')] 
print("num_cols:", num_cols)

#Select column 'Id' from num_cols
num_id=num_cols.pop(0)
print("num_id:", num_id)

#save column 'Id' in num_id variable
num_id=[num_id]
#print num_id
print(num_id)

#Remove column 'label' from numerical columns group
num_cols.remove('label') #label is removed because it's the target to validate the model

#print num_cols variable
print("num_cols:", num_cols)

#count number of numerical and categorical columns in data train
len(num_cols), len(cat_cols)

#Define categorical and nummerical variable in test_data (data test)
#Categorical and numerical variable
#just will select string data type
cat_cols_test = [item[0] for item in test_data.dtypes if item[1].startswith('string')] 
print("cat_cols_test:", cat_cols_test)

#just will select integer or double data type
num_cols_test = [item[0] for item in test_data.dtypes if item[1].startswith('int') | item[1].startswith('double')] 
print("num_cols_test:", num_cols_test)

#Select 'Id' from num_cols_test and save in variable called 'num_id_test'
num_id_test=num_cols_test.pop(0)
print("num_id_test:", num_id_test)

#save num_id_test to list called 'num_id_test'
num_id_test=[num_id_test]
print(num_id_test)
print(num_cols_test)

#count observation in data test
test_data.count()

#count number of numerical and categorical columns in data test
len(num_cols_test), len(cat_cols_test)


#Sample data
#define ratio that want to sample
ratio=0.1 #will take 10% from data

#take 10% sample from data train with replacing false and seed 42 and save in df_sample
df_sample=df_final.sample(False, ratio, 42)

#count observation from df_sample
df_sample.count()

#take 10% sample from data test with replacing false and seed 42 and save in test_sample
test_sample=test_data.sample(False, ratio, 42)

#count observation from test_sample
test_sample.count()


#Check Missing Value in data train
#Check Missing Value in Pyspark Dataframe
def count_nulls(c):
    """Input pyspark dataframe and return list of columns with missing value and it's total value"""
    null_counts = []          #make an empty list to hold our results
    for col in c.dtypes:     #iterate through the column data types we saw above, e.g. ('C0', 'bigint')
        cname = col[0]        #splits out the column name, e.g. 'C0'    
        ctype = col[1]        #splits out the column type, e.g. 'bigint'
        nulls = c.where( c[cname].isNull()).count() #check count of null in column name
        result = tuple([cname, nulls])  #new tuple, (column name, null count)
        null_counts.append(result)      #put the new tuple in our result list
    null_counts=[(x,y) for (x,y) in null_counts if y!=0]  #view just columns that have missing values
    return null_counts

#Call function count_nulls and apply it to data train (df_final)
null_counts = count_nulls(df_final)
null_counts

#From null_counts, we just take information of columns name and save in list "list_cols_miss", like in the script below:
list_cols_miss=[x[0] for x in null_counts]
list_cols_miss

#Create dataframe which just has list_cols_miss
df_miss= df_final.select(*list_cols_miss)
df_miss.dtypes

#Define categorical columns and numerical columns which have missing value.
### for categorical columns
catcolums_miss=[item[0] for item in df_miss.dtypes if item[1].startswith('string')]  #will select name of column with string data type
print("catcolums_miss:", catcolums_miss)

### for numerical columns
numcolumns_miss = [item[0] for item in df_miss.dtypes if item[1].startswith('int') | item[1].startswith('double')] #will select name of column with integer or double data type
print("numcolumns_miss:", numcolumns_miss)


