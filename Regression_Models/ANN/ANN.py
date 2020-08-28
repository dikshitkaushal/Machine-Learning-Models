# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:21:40 2020

@author: -
"""
#PART 1 DATA PREPROCESSING

#import the libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder_x_1=LabelEncoder()
x[:,1]=label_encoder_x_1.fit_transform(x[:,1])
label_encoder_x_2=LabelEncoder()
x[:,2]=label_encoder_x_2.fit_transform(x[:,2])
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
x=x[:,1:]

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature Scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#PART 2 MAKE THE ANN

#import the libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the ANN
classifier=Sequential()

#add the input layer and first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))

#add the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

#add the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to the training set
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

#predicting the test results
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
