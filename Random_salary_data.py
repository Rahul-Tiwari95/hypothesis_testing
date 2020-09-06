# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:25:55 2020

@author: rahul
"""
import pandas as pd
salary_train = pd.read_csv("D:\\ML\\Python\\Python-ML\\Random Forests\\SalaryData_Train.csv")
salary_test = pd.read_csv("D:\\ML\\Python\\Python-ML\\Random Forests\\SalaryData_Test.csv")

colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]

from sklearn.ensemble import RandomForestClassifier
rfsalary = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=15,criterion="entropy")
rfsalary.fit(trainX,trainY) # Error Can not convert a string into float means we have to use LabelEncoder()

# Considering only the string data type columns and 
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
from sklearn import preprocessing
for i in string_columns:
    number = preprocessing.LabelEncoder()
    trainX[i] = number.fit_transform(trainX[i])

rfsalary.fit(trainX,trainY)
# Training Accuracy
trainX["rf_pred"] = rfsalary.predict(trainX)

from sklearn.metrics import confusion_matrix

confusion_matrix(trainY,trainX["rf_pred"]) # Confusion matrix
# Accuracy
print ("Accuracy",(22321+6954)/(22321+332+554+6954)) # 97.06

# Accuracy on testing data 
testX = salary_test[colnames[0:13]]
testY = salary_test[colnames[13]]
# Converting the string values in testing data into float
for i in string_columns:
    number = preprocessing.LabelEncoder()
    testX[i] = number.fit_transform(testX[i])
testX["rf_pred"] = rfsalary.predict(testX)
confusion_matrix(testY,testX["rf_pred"])
# Accuracy 
print ("Accuracy",(10359+2283)/(10359+1001+1417+2283)) # 83.94
