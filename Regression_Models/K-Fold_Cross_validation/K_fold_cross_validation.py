# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:50:56 2020

@author: -
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:08:34 2020

@author: DELL
"""

#import the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#fitting the SVM to our training set
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)

#predicting the new result
y_pred=classifier.predict(x_test)

#using confusion matrix
from sklearn.metrics import confusion_matrix
cmm=confusion_matrix(y_test,y_pred)

#apply K-cross fold Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,x=x_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()

#visulising matrix
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1 , stop = X_set[:, 0].max()+1 , step = 0.01),
                     np.arange(start = X_set[:, 1].min()-1 , stop = X_set[:, 1].max()+1 , step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.65, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i),label = j)
plt.title('SVM Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualising the test set result using graph
from matplotlib.colors import ListedColormap

X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1 , stop = X_set[:, 0].max()+1 , step = 0.01),
                     np.arange(start = X_set[:, 1].min()-1 , stop = X_set[:, 1].max()+1 , step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.65, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                 c=ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
