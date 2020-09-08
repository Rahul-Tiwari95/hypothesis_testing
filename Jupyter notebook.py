#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import pdfplumber
import pandas as pd
from collections import namedtuple

Line = namedtuple('Line','University_name college_id college_name branch_id branch GOPEN_score')

university_re = re.compile(r'Savitribai Phule Pune University')
college_re = re.compile(r'$\d{4} -')
branch_re = re.compile(r'$\d{9} - Civil Engineering')
GOPENS_re = re.compile(r'GOPEN\n{5}') 

file = 'college_cutoffs.pdf'

lines = []

with pdfplumber.open(file) as pdf:
    halfLength = round(len(pdf.pages)/1000)
    pages = pdf.pages[:halfLength]
    for page in pdf.pages:
        text = page.extract_text()
        for line in text.split('\n'):
            univ = university_re.search(line)
            if univ:
                University_name = 'Savitribai Phule Pune University'
            elif line.endswith('Civil Engineering'):
                branch = 'Civil Engineering'
            elif college_re.search(line):
                college_id, college_name = college_re.group(1), college_re.group(2)
                # print('\n')
            elif GOPENS_re.search(line):
                GOPEN_scores = 'GOPEN'
                items = line.split()
                lines.append(Line(University_name,branch,college_id,college_name,GOPEN_scores,*items))


print('Reached')
df = pd.DataRame(lines)
df.head()

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

df = apriori(df, min_support=0.003, max_len=3,use_colnames = True)
df.sort_values('support',ascending = False,inplace=True)

import matplotlib.pyplot as plt

plt.bar(x = list(range(1,11)),height = df.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),df.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(df, metric="lift", min_threshold=1)
rules.sort_values('lift',ascending = False,inplace=True)

