# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:09:33 2022

@author: mmoein2
"""

#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn import neighbors # kNN
from sklearn import metrics #Accuracy metrics
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score #K-fold cross validation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

#Ask for file name and read the file
file_name = 'TSS'
data = pd.read_csv(file_name + '.csv', header=0, index_col = 0)

#Drop empty values
data_clean = data.dropna()

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining X and Y
X = data_clean.drop(columns=['TSS'])
Y = data_clean.Grade

#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) #, stratify=Y)

#Setting up k-Nearest Neighbor and fitting the model with training data
k = 3
#knn = neighbors.KNeighborsClassifier(n_neighbors=k)
knn = neighbors.KNeighborsRegressor(n_neighbors=k)

#Fit final kNN algorithm that uses all training data
knn.fit(X_train, Y_train)

#Run the model on the test (remaining) data and show accuracy
Y_pred = knn.predict(X_test)
print(Y_pred.round(2))
print(Y_test.values)
#score = metrics.accuracy_score(Y_pred, Y_test) 
score = metrics.r2_score(Y_pred, Y_test) 
print('Accuracy score: {0}'.format(score))
