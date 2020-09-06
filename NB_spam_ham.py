# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:12:34 2020

@author: rahul
"""

import pandas as pd
import numpy as np

spam_ham = pd.read_csv("E:\\Data Science\\Data Sheet\\ham_spam.csv",encoding = "ISO-8859-1")

####################### preprocessing: cleaning and splitting data ##########################
import re
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

spam_ham.text = spam_ham.text.apply(cleaning_text)

################## removing empty rows ############################
spam_ham = spam_ham.loc[spam_ham.text != " ",:]

def split_into_words(i):
    return [word for word in i.split(" ")]

################## train and test data sets ############################ 
    
from sklearn.model_selection import train_test_split

spam_train,spam_test = train_test_split(spam_ham,test_size=0.3)

################ converting text into word count matrix format ##################
from sklearn.feature_extraction.text import CountVectorizer

spam_bow = CountVectorizer(analyzer=split_into_words).fit(spam_ham.text)

all_spam_matrix = spam_bow.transform(spam_ham.text)

train_spam_matrix = spam_bow.transform(spam_train.text)

test_spam_matrix = spam_bow.transform(spam_test.text)

from sklearn.naive_bayes import GaussianNB as GB

classifier_gb = GB()
classifier_gb.fit(train_spam_matrix.toarray(),spam_train.type.values) 

train_pred_g = classifier_gb.predict(train_spam_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==spam_train.type) 

test_pred_g = classifier_gb.predict(test_spam_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==spam_test.type) 

#######################################tfidf transformation ################################
# Term Frequency Inverse Document Frequency (tfidf)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(all_spam_matrix)

train_tfidf = tfidf_transformer.transform(train_spam_matrix)
train_tfidf.shape 

test_tfidf = tfidf_transformer.transform(test_spam_matrix)
test_tfidf.shape 

################# Naive Base with tfidf ################################ 

classifier_gb = GB()
classifier_gb.fit(train_tfidf.toarray(),spam_train.type.values) 

train_pred_g = classifier_gb.predict(train_tfidf.toarray())
accuracy_train_g = np.mean(train_pred_g==spam_train.type)
 
test_pred_g = classifier_gb.predict(test_tfidf.toarray())
accuracy_test_g = np.mean(test_pred_g==spam_test.type) 
