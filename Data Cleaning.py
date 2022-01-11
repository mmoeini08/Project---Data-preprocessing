# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:09:33 2022

@author: mmoein2
"""

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

#Ask for file name
#file_name = input("Name of file:")
file_name = 'TSS'
file_header = input("File has labels and header (Y):")

#Create a pandas dataframe from the csv file.      
if file_header == 'Y' or file_header == 'y':
    data = pd.read_csv(file_name + '.csv', header=0, index_col=0) #Remove index_col = 0 if rows do not have headers
else:
    data = pd.read_csv(file_name + '.csv', header=None)

#Remove the TSS column since it is not a numerical variable and it bugs with some imputation techniques. It can be kept with most_frequent, however.
data = data.drop(columns=['TSS'])

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns.values)))
print('')

#Identify missing values
id_missing = input("Identify missing data (Y):")
while id_missing == 'Y' or id_missing == 'y':
    print(data.columns.values)
    column = input("Column:")
    column_missing = data[column].isnull().sum()
    print("{0} has {1} missing values".format(column,column_missing))
    id_missing = input("Identify other missing data (Y):")   

#Simple imputing methods for missing data
impsimp_question = input("Simple impute missing data (Y):")
if impsimp_question == 'Y' or impsimp_question == 'y':
    impsimp_strategy = input("Strategy to impute data: 'mean', 'median', 'most_frequent', or 'constant': ")
    
    if impsimp_strategy == 'constant':
        impsimp_constant = input("Constant value:")
        impsimp_algo = SimpleImputer(missing_values=np.nan, strategy=impsimp_strategy, fill_value=float(impsimp_constant)) #adding float to turn string into a number
    else:
        impsimp_algo = SimpleImputer(missing_values=np.nan, strategy=impsimp_strategy)
   
    impsimp_algo.fit(data)
    impsimp_data = impsimp_algo.transform(data)
    print(impsimp_data)
    
    data_impsimp_clean = pd.DataFrame(data=impsimp_data,
                          columns=data.columns.values,
                          index=data.index)
    data_impsimp_clean.to_csv(file_name + '_impsimp_clean_constant.csv')


#Using kNN to umpute missing data
impkNN_question = input("Use kNN to impute missing data (Y):")
if impkNN_question == 'Y' or impkNN_question == 'y':
    impkNN_number = input("Number of nearest neighbors to use: ")
    impkNN_algo = KNNImputer(missing_values=np.nan, n_neighbors=int(impkNN_number))
   
    impkNN_algo.fit(data)
    impkNN_data = impkNN_algo.transform(data)
    print(impkNN_data)

    data_impkNN_clean = pd.DataFrame(data=impkNN_data,
                          columns=data.columns.values,
                          index=data.index)

    data_impkNN_clean.to_csv(file_name + '_impkNN_clean_6.csv')
