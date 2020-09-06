# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:57:55 2020

@author: rahul
"""

import pandas as pd 
import numpy as np 
import seaborn as sns

forestfires = pd.read_csv("E:\\Data Science\\Data Sheet\\forestfires.csv")
forestfires.head()
forestfires.describe()
forestfires.columns

sns.boxplot(x="size_category",y="temp",data=forestfires,palette = "hls")
sns.boxplot(x="temp",y="size_category",data=forestfires,palette = "hls")
sns.boxplot(x="month",y="temp",data=forestfires,palette = "hls")

sns.pairplot(data=forestfires)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(forestfires,test_size = 0.3)
test.head()
train_X = train.iloc[:,2:30]
train_y = train.iloc[:,30]
test_X  = test.iloc[:,2:30]
test_y  = test.iloc[:,30]


# linear Kernel 
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) 

# poly Kernel 
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) 

# rbf Kernel 
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) 

