# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:24:10 2020

@author: rahul
"""

import pandas as pd
import matplotlib.pylab as plt 
airlines = pd.read_csv("E:\\Data Science\\Data Sheet\\EastWestAirlines.csv")
#normalization
def norm_func(i):
    x = (i-i.min())/(i.max()  -  i.min())
    return (x)

df_norm = norm_func(airlines.iloc[:,:])
df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage method (complete)
z = linkage(df_norm, method="complete",metric="euclidean")
#dendrogram
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=10.,  
)
plt.show()

from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

airlines['clust']=cluster_labels  
airlines = airlines.iloc[:,:]
airlines.head()

airlines.groupby(airlines.clust).mean()

     ################################### K MEANS ############################################
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 

#calculating wss and twss 

k = list(range(2,15))
TWSS = []  
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))     
    
#scree plot to find no of clusters    
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# from scree plot the no of cluster need to be 5

model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_ 
md=pd.Series(model.labels_) 
 
airlines['clust']=md
airlines.head(10) 
airlines = airlines.iloc[:,:]
airlines.groupby(airlines.clust).mean()
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     