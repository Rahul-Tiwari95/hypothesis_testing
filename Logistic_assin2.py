# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:30:16 2020

@author: rahul
"""

import pandas as pd
import seaborn as sb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


bank = pd.read_csv("E:\\Data Science\\Data Sheet\\bank_data.csv")

sb.boxplot(data = bank,orient = "v")
sb.boxplot(x="y",y="balance",data=bank,palette = "hls")
sb.boxplot(x="y",y="duration",data=bank,palette="hls")
sb.boxplot(x="default",y="age",data=bank,palette="hls")
sb.boxplot(x="loan",y="balance",data=bank,palette="hls")


bank_1=list(bank.columns)
bank_1.remove('y')
bank_1
bank_dum=pd.get_dummies(bank[bank_1],drop_first=True)


Y=bank.y
X=bank.iloc[:,0:31]
classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_ 
classifier.predict_proba (X) 


y_pred = classifier.predict(X)
bank["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([bank,y_prob],axis=1)

confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)
