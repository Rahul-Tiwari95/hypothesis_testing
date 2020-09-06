# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 22:22:39 2020

@author: rahul
"""

import pandas as pd 
import numpy as np 
import seaborn as sns

salary_train = pd.read_csv("E:\\Data Science\\Data Sheet\\SalaryData_Train.csv")
salary_test = pd.read_csv("E:\\Data Science\\Data Sheet\\SalaryData_Test.csv")

sns.pairplot(data=salary_train)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(salary_train,test_size = 0.3)
test.head()
train_X = train.iloc[:,0:6]
train_y = train.iloc[:,6]
test_X  = test.iloc[:,0:6]
test_y  = test.iloc[:,6]


# linear Kernel 
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(salary_test.iloc[:,0:6])

np.mean(pred_test_linear==salary_test.iloc[:,6]) 

# poly Kernel 
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(salary_test.iloc[:,0:6])

np.mean(pred_test_poly==salary_test.iloc[:,6]) 

# rbf Kernel 
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(salary_test.iloc[:,0:6])

np.mean(pred_test_rbf==salary_test.iloc[:,6]) 