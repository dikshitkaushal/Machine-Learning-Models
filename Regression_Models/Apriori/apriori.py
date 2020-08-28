# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:04:11 2020

@author: DELL
"""

#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#training apriori on the dataset
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#visualising the results    
results=list(rules)
print(results)