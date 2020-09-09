# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:03:35 2020

@author: rahul
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

groceries = []
with open("E:\\Data Science\\Data Sheet\\groceries.csv") as f:
    groceries = f.read()


groceries = groceries.split("\n")
groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))



one_hot_encoding = TransactionEncoder()
one_hot_groceries= one_hot_encoding.fit(groceries_list).transform(groceries_list) 
one_hot_groceries_df=pd.DataFrame(one_hot_groceries,columns=one_hot_encoding.columns_)  


frequent_itemsets = apriori(one_hot_groceries_df, min_support=0.005, max_len=3,use_colnames = True)

frequent_itemsets.sort_values('support',ascending = False,inplace=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)