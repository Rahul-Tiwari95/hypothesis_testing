# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:18:57 2020

@author: rahul
"""

import pandas as pd

salary_train = pd.read_csv("E:\\Data Science\\Data Sheet\\SalaryData_Train.csv")
salary_test = pd.read_csv("E:\\Data Science\\Data Sheet\\SalaryData_Test.csv")
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
colnames = list(salary_train.columns)
ip_columns = colnames[0:6]
op_column  = colnames[6]


# Spli,tting data into train and test
Xtrain,Xtest,ytrain,ytest = train_test_split(salary_train[ip_columns],salary_train[op_column],test_size=0.3, random_state=0)

ignb = GaussianNB() 

pred_gnb = ignb.fit(Xtrain,ytrain).predict(salary_test.iloc[:,0:6])

confusion_matrix(salary_test.iloc[:,6],pred_gnb) 


