# -*- coding: utf-8 -*-
"""
Created on Sat May 23 02:13:11 2020

@author: -
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:12:45 2020

@author: DELL
"""

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Wine.csv')
x=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#APPLYING PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
explained_variance=pca.explained_variance_ratio_


#fitting the Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#predicting the test set results
y_pred=classifier.predict(x_test)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Visualising the result using graph
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1 , stop = X_set[:, 0].max()+1 , step = 0.01),
                     np.arange(start = X_set[:, 1].min()-1 , stop = X_set[:, 1].max()+1 , step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.65, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green','blue'))(i),label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend()
plt.show()

#Visualising the test set result using graph
from matplotlib.colors import ListedColormap

X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1 , stop = X_set[:, 0].max()+1 , step = 0.01),
                     np.arange(start = X_set[:, 1].min()-1 , stop = X_set[:, 1].max()+1 , step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.65, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                 c=ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend()
plt.show()