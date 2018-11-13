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

#drop column Original_Quote_Date from data test
test_data=test_data.drop('Original_Quote_Date')

#calculate percentage of target and save in dataframe called target_percent
target_percent=df_final.groupBy('label').count().sort(col("count").desc())\
                        .withColumn('total',sum(col('count')).over(window))\
                        .withColumn('Percent',col('count')*100/col('total')) 

#show dataframe terget_percent to check the proportion
target_percent.show()
