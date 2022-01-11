# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:09:33 2022

@author: mmoein2
"""

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
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

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns.values)))
print('')

#Identifying outliers using kNN
outlier_number = input("Number of nearest neighbors to use: ")
outlier_detection = LocalOutlierFactor(n_neighbors=int(outlier_number))
outliers = outlier_detection.fit_predict(data)
print(outliers)
print(outlier_detection.negative_outlier_factor_.round(2)) #Values < 1.5 are found to be outliers

pd.DataFrame(outliers).to_csv("OL.csv", header=["Outliers"], index=False)


