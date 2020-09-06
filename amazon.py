# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:47:02 2020

@author: rahul
"""

import requests   
from bs4 import BeautifulSoup as bs 
import re 

import matplotlib.pyplot as plt
from wordcloud import WordCloud

canon_reviews=[]

for i in range(1,20):
  ca=[]  
  url = "https://www.amazon.in/Canon-1500D-Digital-Camera-S18-55/product-reviews/B07BS4TJ43/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")
  reviews = soup.findAll("span",attrs={"class","a-size-base review-text review-text-content"})
  for i in range(len(reviews)):
    ca.append(reviews[i].getText()) 
  canon_reviews=canon_reviews+ca
  
    
ca_rev_string = " ".join(canon_reviews)

ca_rev_string = re.sub("[^A-Za-z" "]+"," ",ca_rev_string).lower()
ca_rev_string = re.sub("[0-9" "]+"," ",ca_rev_string)

ca_reviews_words = ca_rev_string.split(" ")


with open("E:\\Assingment Excelr\\Text Mining\\stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")


ca_reviews_words = [w for w in ca_reviews_words if not w in stopwords]

ca_rev_string = " ".join(ca_reviews_words)

wordcloud_ca = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ca_rev_string)

plt.imshow(wordcloud_ca)

with open("E:\\Assingment Excelr\\Text Mining\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]

with open("E:\\Assingment Excelr\\Text Mining\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

ca_neg_in_neg = " ".join ([w for w in ca_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ca_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)


ca_pos_in_pos = " ".join ([w for w in ca_reviews_words if w in poswords])


wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ca_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)

