# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:55:34 2020

@author: DELL
"""

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

#using the dendrogram to find optimum number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#training model on our dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

#visualising the clusters
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='cluster1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='green',label='cluster2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='blue',label='cluster3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='cyan',label='cluster4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='magenta',label='cluster5')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income($)')
plt.ylabel('Spending Score(1~100)')
plt.legend()
plt.show()