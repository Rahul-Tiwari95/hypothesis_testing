# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:36:18 2020

@author: rahul
"""

import pandas as pd
import numpy as np

zoo = pd.read_csv("E:\\Data Science\\Data Sheet\\zoo.csv")

from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo,test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)
# Fitting with training data 
neigh.fit(train.iloc[:,1:19],train.iloc[:,0])
# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9]) 
# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9]) 


# 3 to 50 nearest neighbours and storing the accuracy values 
acc = []
for i in range(3,50,1):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:19],train.iloc[:,0])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:19])==train.iloc[:,0])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:19])==test.iloc[:,0])
    acc.append([train_acc,test_acc])


import matplotlib.pyplot as plt 
plt.figure(figsize=(12, 6))
plt.plot(range(3, 50), acc, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accu Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
