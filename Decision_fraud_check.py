# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:19:51 2020

@author: rahul
"""

import pandas as pd

dataset = pd.read_csv("E:\\Data Science\\Data Sheet\\Fraud_check.csv")

dataset.shape
dataset.head()

X = dataset.drop('Urban', axis=1)
y = dataset['Urban']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

