# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 11:40:57 2020

@author: rahul
"""

import pandas as pd

dataset = pd.read_csv("E:\\Data Science\\Data Sheet\\Fraud_check.csv")

X = dataset.drop('Urban', axis=1)
y = dataset['Urban']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
