# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 18:57:15 2020

@author: rahul
"""

import pandas as pd
import seaborn as sb


from sklearn.linear_model import LogisticRegression

credit = pd.read_csv("E:\\Data Science\\Data Sheet\\creditcard.csv")
credit.card.value_counts()

sb.countplot(x="ATTORNEY",data=credit)

sb.boxplot(data = credit,orient = "v")
sb.boxplot(x="card",y="expenditure",data=credit,palette = "hls")
sb.boxplot(x="card",y="income",data=credit,palette="hls")
sb.boxplot(x="owner",y="income",data=credit,palette="hls")
sb.boxplot(x="reports",y="months",data=credit,palette="hls")

Y = credit.card
X = credit.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]


classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_ 
classifier.predict_proba (X) 


y_pred = classifier.predict(X)
credit["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([credit,y_prob],axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)

