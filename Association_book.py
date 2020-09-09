# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:19:48 2020

@author: rahul
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

book= pd.read_csv("E:\\Data Science\\Data Sheet\\book.csv") 



book = apriori(book, min_support=0.003, max_len=3,use_colnames = True)

# Most Frequent item sets based on support 
book.sort_values('support',ascending = False,inplace=True)

import matplotlib.pyplot as plt

plt.bar(x = list(range(1,11)),height = book.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),book.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(book, metric="lift", min_threshold=1)
rules.sort_values('lift',ascending = False,inplace=True)

