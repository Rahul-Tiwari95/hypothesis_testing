# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:42:24 2020

@author: rahul
"""

import pandas as pd

#In this data set the sales coloumn has been changed to categorical variable

dataset = pd.read_csv("E:\\Data Science\\Data Sheet\\Company_Data.csv")

X = dataset.iloc[:,1:6]
y = dataset.iloc[:,11]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
