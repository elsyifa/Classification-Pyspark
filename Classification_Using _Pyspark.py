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

#rename Target to 'label in data train
df = df.withColumnRenamed('QuoteConversion_Flag','label')
#rename Id number ('QuoteNumber') to 'Id' in data train
df = df.withColumnRenamed('QuoteNumber','Id')

#rename Id number ('QuoteNumber') to 'Id' in data test
test_data = test_data.withColumnRenamed('QuoteNumber','Id')

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

#Drop missing value
df_Nomiss=df_final.na.drop()

#fill missing value in categorical variable with most frequent
for x in catcolums_miss:
    mode=df_Nomiss.groupBy(x).count().sort(col("count").desc()).collect()[0][0] #group by based on categories and count each categories and sort descending then take the first value in column
    print(x, mode) #print name of columns and it's most categories 
    df_final = df_final.na.fill({x:mode}) #fill missing value in each columns with most frequent

#fill missing value in numerical variable with average
for i in numcolumns_miss:
    meanvalue = df_final.select(round(mean(i))).collect()[0][0] #calculate average in each numerical column
    print(i, meanvalue) #print name of columns and it's average value
    df_final=df_final.na.fill({i:meanvalue}) #fill missing value in each columns with it's average value
    
#Check Missing value after filling
null_counts = count_nulls(df_final)
null_counts


#Check Missing Value in data test
#We will cleansing missing values in pyspark dataframe.
#Call function to count missing values in test_data
null_test= count_nulls(test_data)
null_test

#take just name of columns that have missing values
list_miss_test=[x[0] for x in null_test]
list_miss_test

#Create dataframe which just has list_cols_miss
test_miss= test_data.select(*list_miss_test)

#view data types in df_miss
test_miss.dtypes

#Define categorical columns and numerical columns which have missing value.
### for categorical columns
catcolums_miss_test=[item[0] for item in test_miss.dtypes if item[1].startswith('string')]  #will select name of column with string data type
print("catcolums_miss_test:", catcolums_miss_test)

### for numerical columns
numcolumns_miss_test = [item[0] for item in test_miss.dtypes if item[1].startswith('int') | item[1].startswith('double')] #will select name of column with integer or double data type
print("numcolumns_miss_test:", numcolumns_miss_test)

#Drop missing value
test_Nomiss=test_data.na.drop()

#fill missing value in categorical variable with most frequent
for x in catcolums_miss_test:
    mode=test_Nomiss.groupBy(x).count().sort(col("count").desc()).collect()[0][0] #group by based on categories and count each categories and sort descending then take the first value in column
    print(x, mode) #print name of columns and it's most categories 
    test_data = test_data.na.fill({x:mode}) #fill missing value in each columns with most frequent

#fill missing value in numerical variable with average
for i in numcolumns_miss_test:
    meanvalue_test = test_data.select(round(mean(i))).collect()[0][0] #calculate average in each numerical column
    print(i, meanvalue_test) #print name of columns and it's average value
    test_data=test_data.na.fill({i:meanvalue_test}) #fill missing value in each columns with it's average value
    
#Check Missing value after filling
%time null_test = count_nulls(test_data)
null_test


#Compare categorical columns in df_final and test_data
#Function to check categorical columns in both data train and data test
def check_category2(a1,a2,y):
    """input are two dataframe you want to compare categorical variables and the colomn category name"""
    print('column:',y)
    #distinct1=a1.select([y]).distinct().count() #count distinct column in dataframe1
    #distinct2=a2.select([y]).distinct().count() #count distinct column in dataframe2
    #if distinct1 == distinct2:
    var1=a1.select([y]).distinct() #define distinct category in column in dataframe1
    var2=a2.select([y]).distinct() #define distinct category in column in dataframe2
    diff2=var2.subtract(var1).collect() #define the different category in dataframe2, return is list
    diff2=[r[y] for r in diff2] #just take the values
    diff1=var1.subtract(var2).collect() #define the different category in dataframe1, return is list
    diff1=[r[y] for r in diff1] #just take the values
    if diff1 == diff2:
        print('diff2:', diff2)
        print('diff1:', diff1)
        print('Columns match!!')
    else:
        if len(diff1)!=0 and len(diff2)==len(diff1):
            print('diff2:', diff2)
            print('diff1:', diff1)
            a2=a2.replace(diff2, diff1, y) #replace the different category in dataframe2 with category in dataframe1
            print('Columns match now!!')
        else:
            if len(diff2)!=len(diff1) and len(diff2)!=0:
                print('diff2:', diff2)
                print('diff1:', diff1)
                dominant1=a1.groupBy(y).count().sort(col("count").desc()).collect()[0][0]
                dominant2=a2.groupBy(y).count().sort(col("count").desc()).collect()[0][0] #define category dominant in dataframe2
                print('dominant2:', dominant2)
                print('dominant1:', dominant1)
                a2=a2.replace(diff2, dominant1, y) #replace different category in dataframe2 with dominant category
                print('Columns match now!!')
            else:     
                print('diff1:', diff1)
                print('diff2:', diff2)
    return a2

#call function to check catgories in data train and test, whether same or not, if not, the different categories will be replaced.
for y in cat_cols_test:
    test_data=check_category2(df_final,test_data,y)
  

#EDA
#Check distribution in each variables
#Pyspark dataframe has limitation in visualization. Then to create visualization we have to convert pyspark dataframe to pandas dataframe.
# convert spark dataframe to pandas for visualization
df_pd=df_final.toPandas()

#Barchart for categorical variable
plt.figure(figsize=(20,10))
plt.subplot(221)
sns.countplot(x='label', data=df_pd, order=df_pd['label'].value_counts().index)
plt.title('TARGET', fontsize=15)
plt.subplot(222)
sns.countplot(y='Field6', data=df_pd, order=df_pd['Field6'].value_counts().index)
plt.title('Field6', fontsize=15)
plt.subplot(223)
sns.countplot(x='Field12', data=df_pd, order=df_pd['Field12'].value_counts().index)
plt.title('Field12', fontsize=15)
plt.show()

#Barchart for categorical variable
plt.figure(figsize=(20,10))
plt.subplot(221)
sns.countplot(y='CoverageField8', data=df_pd, order=df_pd['CoverageField8'].value_counts().index)
plt.title('CoverageField8', fontsize=15)
plt.subplot(222)
sns.countplot(y='CoverageField9', data=df_pd, order=df_pd['CoverageField9'].value_counts().index)
plt.title('CoverageField9', fontsize=15)
plt.subplot(223)
sns.countplot(y='SalesField7', data=df_pd, order=df_pd['SalesField7'].value_counts().index)
plt.title('SalesField7', fontsize=15)
plt.show()

#Categorical vs Target visualization
pd.crosstab(df_pd['Field6'], df_pd['label'], normalize='index').plot.bar(rot=0, stacked=True,
            color=['green', 'red'], figsize=(4,4), title="Field6 VS label")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

pd.crosstab(df_pd['Field12'], df_pd['label'], normalize='index').plot.bar(rot=0, stacked=True,                            
            color=['green', 'red'], figsize=(4,4), title="Field12 VS label")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()

#Numerical Variables
#We have 260 numerical variables, and we will plot just some variables.
#density plot Field7
#plt.figure(figsize=(24,5))
sns.distplot(df_pd['Field7'])
plt.show()

#Numerical vs Target visualization
#show distribution 'Field7' vs 'label'
#plt.figure(figsize=(20,8))
sns.kdeplot(df_pd[df_pd["label"]==0]["Field7"], label="0", color="green")
sns.kdeplot(df_pd[df_pd["label"]==1]["Field7"], label="1", color="red")
plt.title("Field7 VS label")
plt.show()

#Check outlier in numerical variable
df_pd[["Field7"]].boxplot(sym='g-*', grid=True)
plt.show()


#Insignificant Categories in Data train
#Define the threshold for insignificant categories
threshold=98
threshold2=0.7

#function to replace insignificant categories in data train
def replace_cat2(f,cols):
    """input are dataframe and categorical variables, replace insignificant categories (percentage <=0.7) with largest number
    of catgories and output is new dataframe """
    df_percent=f.groupBy(cols).count().sort(col("count").desc())\
                .withColumn('total',sum(col('count')).over(window))\
                .withColumn('Percent',col('count')*100/col('total')) #calculate the percentage-save in Percent columns from each categories
    dominant_cat=df_percent.select(df_percent['Percent']).collect()[0][0] #calculate the highest percentage of category
    count_dist=f.select([cols]).distinct().count() #calculate distinct values in that columns
    if count_dist > 2 and dominant_cat <= threshold :
        print('column:', cols)
        cols_names.append(cols)  #combine with previous list
        replacement=f.groupBy(cols).count().sort(col("count").desc()).collect()[0][0] #define dominant category 
        print("replacement:",replacement)
        replacing.append(replacement) #combine with previous list
        insign_cat=df_percent.filter(df_percent['Percent']< threshold2).select(df_percent[cols]).collect() #calculate insignificant categories
        insign_cat=[r[cols] for r in insign_cat] #just take the values
        category.append(insign_cat) #combine with previous list
        print("insign_cat:",insign_cat)
        f=f.replace(insign_cat,replacement, cols) #replace insignificant categories with dominant categories
    return f

#call function replacing insignificant categories in data train
replacing=[]
cols_names=[]
category=[]
for cols in cat_cols:
    df_final=replace_cat2(df_final,cols)

#check length in list cols_names, category and replacing
len(cols_names), len(category), len(replacing)

#Create dataframe of replaced categories
g=spark.createDataFrame(list(zip(cols_names, replacing, category)),['cols_names', 'replacing', 'category'])
g.show(9)

#Replacing Insignificant Categories in data test
#We already have a dataframe containing any categories that need to be replaced, 
#we got it when the process of replacing the insignificant categories in the data train, the data frame is called g. 
#Based on those information, insignificant categories on data test will be replaced.
cols_names_list=g.select('cols_names').collect() #select just cols_names from dataframe g
cols_names_list=[r['cols_names'] for r in cols_names_list] #take just the values

#function to replace insignificant categories in data test
for z in cols_names_list:
    print('cols_names:',z)
    replacement_cat=g.filter(g['cols_names']== z).select(g['replacing']).collect()[0][0] #select values of replacing columns accoring to z in cols_names 
    print('replacement_cat:', replacement_cat)
    insignificant_cat=g.filter(g['cols_names']== z).select(g['category']).collect()[0][0] #select values of category columns accoring to z in cols_names
    print('insignificant_cat:',insignificant_cat)
    test_data=test_data.replace(insignificant_cat,replacement_cat, z) #replace insignificant cat with replacement value
    
#Handle of outlier in data train
#Calculate Upper&Lower side in pandas dataframe
df_describe=df_pd.describe()
df_describe

#Calculate Upper&Lower side in pyspark dataframe
#create quantile dataframe
def quantile(e):
    """Input is dataframe and return new dataframe with value of quantile from numerical columns"""
    percentiles = [0.25, 0.5, 0.75]
    quant=spark.createDataFrame(zip(percentiles, *e.approxQuantile(num_cols, percentiles, 0.0)),
                               ['percentile']+num_cols) #calculate quantile from pyspark dataframe, 0.0 is relativeError,
                                                        #The relative target precision to achieve (>= 0). If set to zero, 
                                                        #the exact quantiles are computed, which could be very expensive
                                                        #and aggregate the result with percentiles variable, 
                                                        #then create pyspark dataframe
    return quant

#call quantile function 
%time quantile=quantile(df_sample)

#function to calculate uppler side
def upper_value(b,c):
    """Input is quantile dataframe and name of numerical column and Retrun upper value from the column"""
    q1 = b.select(c).collect()[0][0] #select value of q1 from the column
    q2 = b.select(c).collect()[1][0] #select value of q2 from the column
    q3 = b.select(c).collect()[2][0] #select value of q3 from the column
    IQR=q3-q1  #calculate the value of IQR
    upper= q3 + (IQR*1.5)   #calculate the value of upper side
    return upper

#function to calculate lower side
def lower_value(b,c):
    """Input is quantile dataframe and name of numerical column and Retrun lower value from the column"""
    q1 = b.select(c).collect()[0][0] #select value of q1 from the column
    q2 = b.select(c).collect()[1][0] #select value of q2 from the column
    q3 = b.select(c).collect()[2][0] #select value of q3 from the column
    IQR=q3-q1                   #calculate the value of IQR
    lower= q1 - (IQR*1.5)       #calculate the value of lower side
    return lower

#function for replacing outlier by upper side
def replce_outlier_up2(d,col, value):
    """Input is name of numerical column and it's upper side value"""
    d=d.withColumn(col, F.when(d[col] > value , value).otherwise(d[col]))
    return d

#function for replacing outlier with lower side
def replce_outlier_low2(d,col, value):
    """Input is name of numerical column and it's lower side value"""
    d=d.withColumn(col, F.when(d[col] < value , value).otherwise(d[col]))
    return d

#call function to calculate lower side and replace value under lower side with value lower side at all numerical variables
for i in num_cols:
    lower=lower_value(quantile,i)
    df_final=replce_outlier_low2(df_final, i, lower)
    
#call function to calculate upper side and replace value above upper side with value upper side at all numerical variables
for x in num_cols:
    upper=upper_value(quantile,x)
    df_final=replce_outlier_up2(df_final, x, upper)
    
#Handle of outlier in data test
#create quantile dataframe
def quantile(e):
    """Input is dataframe and return new dataframe with value of quantile from numerical columns"""
    percentiles = [0.25, 0.5, 0.75]
    quant=spark.createDataFrame(zip(percentiles, *e.approxQuantile(num_cols_test, percentiles, 0.0)),
                               ['percentile']+num_cols_test) #calculate quantile from pyspark dataframe, 0.0 is relativeError,
                                                        #The relative target precision to achieve (>= 0). If set to zero, 
                                                        #the exact quantiles are computed, which could be very expensive
                                                        #and aggregate the result with percentiles variable, 
                                                        #then create pyspark dataframe
    return quant

#call funtion quantile
quantile=quantile(test_sample)

#call function to calculate lower side and replace value under lower side with value lower side at all numerical variables
for i in num_cols_test:
    lower=lower_value(quantile,i)
    test_data=replce_outlier_low2(test_data, i, lower)
    
#call function to calculate upper side and replace value above upper side with value upper side at all numerical variables
for x in num_cols_test:
    upper=upper_value(quantile,x)
    test_data=replce_outlier_up2(test_data, x, upper)
    
#Feature Engineering
#function to check distinct categories in data train and data test
def check_distinct(a1,a2):
    """input are two dataframe that you want to compare categorical variables and the output is 
    total distinct categories in both dataframe"""
    total1=0
    total2=0
    for y in cat_cols:
        distinct1=a1.select([y]).distinct().count() #count distinct column in dataframe1
        distinct2=a2.select([y]).distinct().count() #count distinct column in dataframe2
        var1=a1.select([y]).distinct().collect() #define distinct category in column in dataframe1
        var1=[r[y] for r in var1]
        var2=a2.select([y]).distinct().collect()
        var2=[r[y] for r in var2]
        total1=total1+distinct1
        total2=total2+distinct2   
    return total1, total2  

#function to execute feature engineering
def feature_engineering(a1):    
    """Function for feature engineering (StringIndexer and OneHotEncoder process)"""
    cat_columns_string_vec = []
    for c in cat_cols:
        cat_columns_string= c+"_vec"
        cat_columns_string_vec.append(cat_columns_string)
    stringIndexer = [StringIndexer(inputCol=x, outputCol=x+"_Index")
                  for x in cat_cols]
    #use oneHotEncoder to convert categorical variable to binary
    encoder = [OneHotEncoder(inputCol=x+"_Index", outputCol=y)
           for x,y in zip(cat_cols, cat_columns_string_vec)]
    #create list of stringIndexer and encoder with 2 dimension
    tmp = [[i,j] for i,j in zip(stringIndexer, encoder)]
    tmp = [i for sublist in tmp for i in sublist]
    cols_assember=num_id + num_cols + cat_columns_string_vec
    assembler=VectorAssembler(inputCols=cols_assember, outputCol='features')
    tmp += [assembler]
    pipeline=Pipeline(stages=tmp)
    df_final_feat=pipeline.fit(a1).transform(a1)
    return df_final_feat

#fucntion to call fucntion feature_engineering and check_distinct
def Main_feature_engineering(df,df2): 
    """Function for calling check_distinct and feature_engineering. Then Join data train and data test if distinct categories 
    between data train and data test not same then do feature engineering, If distinct same will do feature engineering data train
    and data test separately"""
    dist_total1, dist_total2=check_distinct(df,df2)   
    if dist_total1!=dist_total2:
        Label_df=df.select('Id', 'label')
        df_final2=df.drop('label')
        all_df =df_final2.union(df2)
        all_df_feat=feature_engineering(all_df)
        id_train=df.select('Id').collect()
        id_train=[r['Id'] for r in id_train]
        id_test=df2.select('Id').collect()
        id_test=[r['Id'] for r in id_test]
        a=all_df_feat.filter(all_df['Id'].isin(id_train))
        b=all_df_feat.filter(all_df['Id'].isin(id_test))
        a=a.join(Label_df, 'Id')
    else:
        a=feature_engineering(df)
        b=feature_engineering(df2)        
    return a,b

#call function feature engineering
%time data2, test2=Main_feature_engineering(df_final, test_data)

#view result of feature engineering in data train
data2.select('Id', 'features').show(5)

#view result of feature engineering in data test
test2.select('Id', 'features').show(5)

#Split Data train to train and test
#Split df_final to train and test, train 70% and test 30%. Define seed 24 so the random data that we split will not change.
#we can define seed with any value
data_train, data_test=data2.randomSplit([0.7,0.3], 24)


#Modelling & Evaluation
#Logistic Regression
#Create logistic regression model to data train
lr=LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(data_train)

#Transform model to data test
lr_result = lr_model.transform(data_test)

#view id, label, prediction and probability from result of modelling
lr_result.select('Id', 'label', 'prediction', 'probability').show(5)

#Logistic Regression Evaluation
#Evaluate model by checking accuracy and AUC value
lr_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
lr_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
lr_AUC  = lr_eval.evaluate(lr_result)
lr_ACC  = lr_eval2.evaluate(lr_result, {lr_eval2.metricName:"accuracy"})

print("Logistic Regression Performance Measure")
print("Accuracy = %0.2f" % lr_ACC)
print("AUC = %.2f" % lr_AUC)

#ROC Grafik
#Create ROC grafik from lr_result
PredAndLabels           = lr_result.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Area under ROC
print("Logistic Regression Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

#confusion Matrix
cm_lr_result = lr_result.crosstab("prediction", "label")
cm_lr_result = cm_lr_result.toPandas()
cm_lr_result

#calculate Accuracy, Sensitivity, Specificity, Precision
TP = cm_lr_result["1"][0]
FP = cm_lr_result["0"][0]
TN = cm_lr_result["0"][1]
FN = cm_lr_result["1"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

#Calculate Gini Coefficient from AUC
AUC = lr_AUC
Gini = (2 * AUC - 1)
print("AUC=%.2f" % AUC)
print("GINI ~=%.2f" % Gini)

#Calculate Log Loss in pandas dataframe
#Create Dataframe to Calculate Log Loss
y_test= data_test.select('label')
lr_proba=lr_result.select('probability')

#Convert lr_probaspark dataframe to numpy array
lr_proba= np.array(lr_result.select('probability').collect())

#Convert numpy array 3 dimentional to 2 dimentional
lr_proba=lr_proba.reshape(-1, lr_proba.shape[-1])

#Convert y_test dataframe to pandas dataframe
y_test=y_test.toPandas()

#Convert y_test pandas dataframe to pandas series
y_test=pd.Series(y_test['label'].values)

#Calculate log loss from logistic regression
LogLoss = log_loss(y_test, lr_proba) 

print("Log Loss Linear Regression:%.4f" % LogLoss)

#Logistic Regression With Hyper-Parameter Tuning
#define logistic regression model
lr_hyper=LogisticRegression(featuresCol='features', labelCol='label')


#Hyper-Parameter Tuning
paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr_hyper.regParam, [0.1, 0.01]) \
    .addGrid(lr_hyper.elasticNetParam, [0.8, 0.7]) \
    .build()
crossval_lr = CrossValidator(estimator=lr_hyper,
                             estimatorParamMaps=paramGrid_lr,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=3)
#fit model to data train
lr_model_hyper = crossval_lr.fit(data_train)

#Transform model to data test
lr_result_hyper = lr_model_hyper.transform(data_test)

#view id, label, prediction and probability from result of modelling
lr_result_hyper.select('Id', 'label', 'prediction', 'probability').show(5)

#Logistic Regression With Hyper-Parameter Tuning Evaluation
#Evaluate model by checking accuracy and AUC value
lr_hyper_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
lr_hyper_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
lr_hyper_AUC  = lr_hyper_eval.evaluate(lr_result_hyper)
lr_hyper_ACC  = lr_hyper_eval2.evaluate(lr_result_hyper, {lr_hyper_eval2.metricName:"accuracy"})

print("Logistic Regression Performance Measure")
print("Accuracy = %0.2f" % lr_hyper_ACC)
print("AUC = %.2f" % lr_hyper_AUC)

#ROC Grafik
PredAndLabels           = lr_result_hyper.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Area under ROC
print("Logistic Regression Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

#confusion matrix
cm_lr_result_hyper = lr_result_hyper.crosstab("prediction", "label")
cm_lr_result_hyper = cm_lr_result_hyper.toPandas()
cm_lr_result_hyper

#calculate Accuracy, Sensitivity, Specificity, Precision
TP = cm_lr_result_hyper["1"][0]
FP = cm_lr_result_hyper["0"][0]
TN = cm_lr_result_hyper["0"][1]
FN = cm_lr_result_hyper["1"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

#Calculate Gini Coefisient from AUC
AUC = lr_hyper_AUC
Gini_lr_hyper = (2 * AUC - 1)
print("AUC=%.2f" % AUC)
print("GINI ~=%.2f" % Gini_lr_hyper)

#Calculate Log Loss in pandas dataframe
#Create Dataframe to Calculate Log Loss
y_test= titanic_test.select('label')
lr_hyper_proba=lr_result_hyper.select('probability')

#Convert lr_probaspark dataframe to numpy array
lr_hyper_proba= np.array(lr_hyper_proba.select('probability').collect())

#Convert numpy array 3 dimentional to 2 dimentional
lr_hyper_proba=lr_hyper_proba.reshape(-1, lr_hyper_proba.shape[-1])

#Convert y_test dataframe to pandas dataframe
y_test=y_test.toPandas()

#Convert y_test pandas dataframe to pandas series
y_test=pd.Series(y_test['label'].values)

#Calculate log loss from logistic regression hyper parameter
LogLoss = log_loss(y_test, lr_hyper_proba) 

print("Log Loss Linear Regression:%.4f" % LogLoss)


#Decision Tree
#Create decision tree model to data train
dt=DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dt_model = dt.fit(data_train)

##Transform model to data test
dt_result = dt_model.transform(data_test)

#view id, label, prediction and probability from result of modelling
dt_result.select('Id', 'label', 'prediction', 'probability').show(5)

# Decision Tree Evaluation
#Evaluate model by calculating accuracy and area under curve (AUC)
dt_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
dt_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
dt_AUC  = dt_eval.evaluate(dt_result)
dt_ACC  = dt_eval2.evaluate(dt_result, {dt_eval2.metricName:"accuracy"})

print("Decision Tree Performance Measure")
print("Accuracy = %0.2f" % dt_ACC)
print("AUC = %.2f" % dt_AUC)

#ROC Grafik
PredAndLabels           = dt_result.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Area under ROC
print("Decision Tree Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc="lower right")
plt.show()

#confusion matrix
cm_dt_result = dt_result.crosstab("prediction", "label")
cm_dt_result = cm_dt_result.toPandas()
cm_dt_result

#calculate accuracy, sensitivity, specificity and precision
TP = cm_dt_result["1"][0]
FP = cm_dt_result["0"][0]
TN = cm_dt_result["0"][1]
FN = cm_dt_result["1"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

#Calculate Gini Coeffiecient from AUC
AUC = dt_AUC
Gini_dt = (2 * AUC - 1)
print("AUC=%.2f" % AUC)
print("GINI ~=%.2f" % Gini_dt)

#Calculate Log Loss in pandas dataframe
#Create Dataframe to Calculate Log Loss
y_test= data_test.select('label')
dt_proba=dt_result.select('probability')

##Convert lr_probaspark dataframe to numpy array
dt_proba= np.array(dt_proba.select('probability').collect())

#Convert numpy array 3 dimentional to 2 dimentional
dt_proba=dt_proba.reshape(-1, dt_proba.shape[-1])

#Convert y_test dataframe to pandas dataframe
y_test=y_test.toPandas()

#Convert y_test pandas dataframe to pandas series
y_test=pd.Series(y_test['label'].values)

#Calculate log loss from Decision Tree
LogLoss = log_loss(y_test, dt_proba) 

print("Log Loss Decision Tree:%.4f" % LogLoss)

#Decision Tree With Hyper-Parameter Tuning
#define decision tree model
dt_hyper=DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', impurity='gini')

#Hyper-Parameter Tuning
paramGrid_dt = ParamGridBuilder() \
    .addGrid(dt_hyper.maxDepth, [5, 7]) \
    .addGrid(dt_hyper.maxBins, [10,20]) \
    .build()
crossval_dt = CrossValidator(estimator=dt_hyper,
                             estimatorParamMaps=paramGrid_dt,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=5)
#fit model to data train
dt_model_hyper = crossval_dt.fit(data_train)

#transform model to data test
dt_result_hyper = dt_model_hyper.transform(data_test)

#view id, label, prediction and probability from result of modelling 
dt_result_hyper.select('Id', 'label', 'prediction', 'probability').show(5)

#Decision Tree With Hyper-Parameter Tuning Evaluation
#Evaluate model by calculating accuracy and area under curve (AUC)
dt_hyper_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
dt_hyper_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
dt_hyper_AUC  = dt_hyper_eval.evaluate(dt_result_hyper)
dt_hyper_ACC  = dt_hyper_eval2.evaluate(dt_result_hyper, {dt_hyper_eval2.metricName:"accuracy"})

print("Decision Tree Performance Measure")
print("Accuracy = %0.2f" % dt_hyper_ACC)
print("AUC = %.2f" % dt_hyper_AUC)

#ROC Grafik
PredAndLabels           = dt_result_hyper.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Area under ROC
print("Decision Tree Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc="lower right")
plt.show()

#Confusion Matrix
cm_dt_result_hyper = dt_result_hyper.crosstab("prediction", "label")
cm_dt_result_hyper = cm_dt_result_hyper.toPandas()
cm_dt_result_hyper

#calculate accuracy, sensitivity, specificity and precision
TP = cm_dt_result_hyper["1"][0]
FP = cm_dt_result_hyper["0"][0]
TN = cm_dt_result_hyper["0"][1]
FN = cm_dt_result_hyper["1"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

#Calculate Gini Coefficient from AUC
AUC = dt_hyper_AUC
Gini_dt_hyper= (2 * AUC -1)

print("AUC=%.2f" % AUC)
print("GINI ~=%.2f" % Gini_dt_hyper)

#Calculate Log Loss in pandas dataframe
#Create Dataframe to Calculate Log Loss
y_test= data_test.select('label')
dt_hyper_proba=dt_result_hyper.select('probability')

#Convert lr_probaspark dataframe to numpy array
dt_hyper_proba= np.array(dt_hyper_proba.select('probability').collect())

#Convert numpy array 3 dimentional to 2 dimentional
dt_hyper_proba=dt_hyper_proba.reshape(-1, dt_hyper_proba.shape[-1])

#Convert y_test dataframe to pandas dataframe
y_test=y_test.toPandas()

#Convert y_test pandas dataframe to pandas series
y_test=pd.Series(y_test['label'].values)

#Calculate log loss from Decision Tree hyper parameter
LogLoss = log_loss(y_test, dt_hyper_proba) 

print("Log Loss Decision Tree:%.4f" % LogLoss)

#Random Forest
#Create decision tree model to data train
rf = RandomForestClassifier(featuresCol='features', labelCol="label")
rf_model = rf.fit(data_train)

#transform model to data test
rf_result = rf_model.transform(data_test)

#view id, label, prediction and probability from result of modelling
rf_result.select('Id', 'label', 'prediction', 'probability').show(5)

#Random Forest Evaluation
#Evaluate model by calculatin accuracy and area under curve (AUC)
rf_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
rf_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
rf_AUC  = rf_eval.evaluate(rf_result)
rf_ACC  = rf_eval2.evaluate(rf_result, {rf_eval2.metricName:"accuracy"})

print("Decision Tree Performance Measure")
print("Accuracy = %0.2f" % rf_ACC)
print("AUC = %.2f" % rf_AUC)

#ROC Grafik
PredAndLabels           = rf_result.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Area under ROC
print("Random Forest Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.show()

#Confusion Matrix
cm_rf_result = rf_result.crosstab("prediction", "label")
cm_rf_result = cm_rf_result.toPandas()
cm_rf_result

#calculate accurary,sensitivity, specificity and precision 
TP = cm_rf_result["1"][0]
FP = cm_rf_result["0"][0]
TN = cm_rf_result["0"][1]
FN = cm_rf_result["1"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

#Calculate Gini Coefficient from AUC
AUC = rf_AUC
Gini_rf= (2 * AUC -1)

print("AUC=%.2f" % AUC)
print("GINI ~=%.2f" % Gini_rf)

#Calculate Log Loss in pandas dataframe
#Create Dataframe to Calculate Log Loss
y_test= data_test.select('label')
rf_proba=rf_result.select('probability')

#Convert rf_probaspark dataframe to numpy array
rf_proba= np.array(rf_proba.select('probability').collect())

#Convert numpy array 3 dimentional to 2 dimentional
rf_proba=rf_proba.reshape(-1, rf_proba.shape[-1])

#Convert y_test dataframe to pandas dataframe
y_test=y_test.toPandas()

#Convert y_test pandas dataframe to pandas series
y_test=pd.Series(y_test['label'].values)

#Calculate log loss from Random Forest
LogLoss = log_loss(y_test, rf_proba) 

print("Log Loss Random Forest:%.4f" % LogLoss)

#Random Forest With Hyper-Parameter
#define random forest model
rf_hyper= RandomForestClassifier(featuresCol='features', labelCol="label")

# Hyper-Parameter Tuning
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf_hyper.numTrees, [40, 60, 80, 100]) \
    .build()
crossval_rf = CrossValidator(estimator=rf_hyper,
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=3) 
#fit model to data train
rf_model_hyper=crossval_rf.fit(data_train)

#transfrom model to data test
rf_result_hyper = rf_model_hyper.transform(data_test)

#view id, label, prediction and probability from result of modelling
rf_result_hyper.select('Id', 'label', 'prediction', 'probability').show(5)

#Random Forest With Hyper-Parameter Evaluation
#Evaluate model by calculating accuracy and area under curve (AUC)
rf_hyper_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
rf_hyper_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
rf_hyper_AUC  = rf_hyper_eval.evaluate(rf_result_hyper)
rf_hyper_ACC  = rf_hyper_eval2.evaluate(rf_result_hyper, {rf_hyper_eval2.metricName:"accuracy"})

print("Decision Tree Performance Measure")
print("Accuracy = %0.2f" % rf_hyper_ACC)
print("AUC = %.2f" % rf_hyper_AUC)

#ROC Grafik
PredAndLabels           = rf_result_hyper.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Area under ROC
print("Random Forest Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.show()

#Confusion Matrix
cm_rf_result_hyper = rf_result_hyper.crosstab("prediction", "label")
cm_rf_result_hyper = cm_rf_result_hyper.toPandas()
cm_rf_result_hyper

#calculate accuracy, sensitivity, specificity and precision
TP = cm_rf_result_hyper["1"][0]
FP = cm_rf_result_hyper["0"][0]
TN = cm_rf_result_hyper["0"][1]
FN = cm_rf_result_hyper["1"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

#Calculate Gini Coefficient from AUC
AUC = rf_hyper_AUC
Gini_rf_hyper= (2 * AUC -1)

print("AUC=%.2f" % AUC)
print("GINI ~=%.2f" % Gini_rf_hyper)

#Calculate Log Loss in pandas dataframe
#Create Dataframe to Calculate Log Loss
y_test= data_test.select('label')
rf_hyper_proba=rf_result_hyper.select('probability')

#Convert pyspark dataframe to numpy array
rf_hyper_proba= np.array(rf_hyper_proba.select('probability').collect())

#Convert numpy array 3 dimentional to 2 dimentional
rf_hyper_proba=rf_hyper_proba.reshape(-1, rf_hyper_proba.shape[-1])

#Convert y_test dataframe to pandas dataframe
y_test=y_test.toPandas()

#Convert y_test pandas dataframe to pandas series
y_test=pd.Series(y_test['label'].values)

#Calculate log loss from Random Forest hyper parameter
LogLoss = log_loss(y_test, rf_hyper_proba) 

print("Log Loss Random Forest:%.4f" % LogLoss)

#Gradient Boosting
#create gradient boosting model in data train
gbt = GBTClassifier(featuresCol="features", labelCol="label",  maxIter=10)
gbt_model = gbt.fit(data_train)

#transfrom model to data test
gbt_result = gbt_model.transform(data_test)

#view id, label, prediction and probability from result of modelling
gbt_result.select('Id', 'label', 'prediction', 'probability').show(5)

#Gradient Boosting Evaluation
#Evaluate model by calculating accuracy and area under curve (AUC)
gbt_eval = BinaryClassificationEvaluator(rawPredictionCol="probability",labelCol="label")
gbt_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
gbt_AUC  = gbt_eval.evaluate(gbt_result)
gbt_ACC  = gbt_eval2.evaluate(gbt_result, {gbt_eval2.metricName:"accuracy"})

print("Gradient Boosted Tree Performance Measure")
print("Accuracy = %0.2f" % gbt_ACC)
print("AUC = %.2f" % gbt_AUC)

#ROC Grafik
PredAndLabels           = gbt_result.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Area under ROC
print("Gradient Boosting Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting')
plt.legend(loc="lower right")
plt.show()

#Confusion Matrix
cm_gbt_result = gbt_result.crosstab("prediction", "label")
cm_gbt_result = cm_gbt_result.toPandas()
cm_gbt_result

#calculate accuracy, sensitivity, specificity and precision
TP = cm_gbt_result["1"][0]
FP = cm_gbt_result["0"][0]
TN = cm_gbt_result["0"][1]
FN = cm_gbt_result["1"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

#Calculate Gini Coefficient from AUC
AUC = gbt_AUC
Gini_gbt= (2 * AUC -1)

print("AUC=%.2f" % AUC)
print("GINI ~=%.2f" % Gini_gbt)

#Calculate Log Loss in pandas dataframe
#Create Dataframe to Calculate Log Loss
y_test= data_test.select('label')
gbt_proba=gbt_result.select('probability')

#Convert pyspark dataframe to numpy array
gbt_proba= np.array(gbt_proba.select('probability').collect())

#Convert numpy array 3 dimentional to 2 dimentional
gbt_proba=gbt_proba.reshape(-1, gbt_proba.shape[-1])

#Convert y_test dataframe to pandas dataframe
y_test=y_test.toPandas()

#Convert y_test pandas dataframe to pandas series
y_test=pd.Series(y_test['label'].values)

#Calculate log loss from Gradient Boosting
LogLoss = log_loss(y_test, gbt_proba) 

print("Log Loss Gradient Boosting:%.4f" % LogLoss)

#Gradient Boosting With Hyper-Parameter
#define gradient boosting model
gbt_hyper= GBTClassifier(featuresCol="features", labelCol="label")

# Hyper-Parameter Tuning
paramGrid_gbt = ParamGridBuilder() \
    .addGrid(gbt_hyper.maxIter, [10])\
    .addGrid(gbt_hyper.maxDepth, [6, 7,10]) \
    .build()
crossval_gbt = CrossValidator(estimator=gbt_hyper,
                             estimatorParamMaps=paramGrid_gbt,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=3)
#fit model to data train
gbt_model_hyper = crossval_gbt.fit(data_train)

#transfrom model to data test
gbt_result_hyper = gbt_model_hyper.transform(data_test)

#view id, label, prediction and probability from result of modelling
gbt_result_hyper.select('Id', 'label', 'prediction', 'probability').show(5)

#Gradient Boosting With Hyper-Parameter Evaluation
#Evaluate model by calculating accuracy and area under curve (AUC)
gbt_eval_hyper = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
gbt_eval_hyper2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
gbt_hyper_AUC  = gbt_eval_hyper.evaluate(gbt_result_hyper)
gbt_hyper_ACC  = gbt_eval_hyper2.evaluate(gbt_result_hyper, {gbt_eval_hyper2.metricName:"accuracy"})


print("Gradient Boosted Tree Performance Measure")
print("Accuracy = %0.2f" % gbt_hyper_ACC)
print("AUC = %.2f" % gbt_hyper_AUC)

#ROC Grafik
PredAndLabels           = gbt_result_hyper.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Area under ROC
print("Gradient Boosting Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting')
plt.legend(loc="lower right")
plt.show()

#confusion Matrix
cm_gbt_result_hyper = gbt_result_hyper.crosstab("prediction", "label")
cm_gbt_result_hyper = cm_gbt_result_hyper.toPandas()
cm_gbt_result_hyper

#calculate accuracy, sensitivity, specificity and precision
TP = cm_gbt_result_hyper["1"][0]
FP = cm_gbt_result_hyper["0"][0]
TN = cm_gbt_result_hyper["0"][1]
FN = cm_gbt_result_hyper["1"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

#Calculate Gini Coefficient from AUC
AUC = gbt_hyper_AUC
Gini_gbt_hyper= (2 * AUC -1)

print("AUC=%.2f" % AUC)
print("GINI ~=%.2f" % Gini_gbt_hyper)

#Calculate Log Loss in pandas dataframe
#Create Dataframe to Calculate Log Loss
y_test= data_test.select('label')
gbt_hyper_proba=gbt_result_hyper.select('probability')

#Convert pyspark dataframe to numpy array
gbt_hyper_proba= np.array(gbt_hyper_proba.select('probability').collect())

#Convert numpy array 3 dimentional to 2 dimentional
gbt_hyper_proba=gbt_hyper_proba.reshape(-1, gbt_hyper_proba.shape[-1])

#Convert y_test dataframe to pandas dataframe
y_test=y_test.toPandas()

#Convert y_test pandas dataframe to pandas series
y_test=pd.Series(y_test['label'].values)

#Calculate log loss from Gradient Boosting hyper parameter
LogLoss = log_loss(y_test, gbt_hyper_proba) 

print("Log Loss Gradient Boosting:%.4f" % LogLoss)


#Implementation Modelling to data test
#Prediction using Logistic Regression
#transform logistic regression to data test
lr_predict = lr_model.transform(test2)

#view id, label, prediction and probability from result of modelling
lr_predict.select('Id', 'prediction', 'probability').show(5)

#select id and prediction from result of modelling and save in data frame called my_submission
my_submission=lr_predict.select("Id","prediction")

#convert to Pandas dataframe
my_submission=my_submission.toPandas()

#save to csv
my_submission.to_csv('E:/Datalabs/Classification/Home_Quote_conversion/my_submission.csv', index = False, header = True)


#Prediction using Gradient Boosting
#transfrom gradient boosting model to data test
gbt_predict = gbt_model.transform(test_data_feat)

#view id, label, prediction and probability from result of modelling
gbt_predict.select('Id', 'prediction', 'probability').show(5)

#select id and prediction from result of modelling and save in data frame called my_submission
my_submission2=gbt_predict.select("Id","prediction")

#convert to Pandas dataframe
my_submission2=my_submission2.toPandas()

#save to csv
my_submission2.to_csv('E:/Datalabs/Classification/Home_Quote_conversion/my_submission2.csv', index = False, header = True)
