# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:49:38 2020

@author: DELL
"""

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#cleaning the text
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    ps=PorterStemmer()
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review = review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values;

#splitting the dataset into train_test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#fitting naive bais to the model
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#predicting the result
y_pred=classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
