# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:59:19 2020

@author: rahul
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

movies = pd.read_csv("E:\\Data Science\\Data Sheet\\my_movies.csv") 



movies = apriori(movies, min_support=0.005, max_len=3,use_colnames = True)

# Most Frequent item sets based on support 
movies.sort_values('support',ascending = False,inplace=True)

import matplotlib.pyplot as plt

plt.bar(x = list(range(1,11)),height = movies.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),movies.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(movies, metric="lift", min_threshold=1)
rules.sort_values('lift',ascending = False,inplace=True)
