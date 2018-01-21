#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 08:56:44 2017

@author: saurabh
"""

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt;
from sklearn import model_selection
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


names=['sepal-length','sepal-width','petal-length','petal-width','class']
df=pd.read_csv('irisData.csv',names=names)


print(df.head(20))
print(df.shape)
print(df.describe())
print(df.groupby('class').size())

#figure(0)
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

df.hist()
plt.show()

scatter_matrix(df)
plt.show()

arr=np.array(df.values)
X=np.array(arr[:,0:4])
y=np.array(arr[:,4])
print(y.shape)
val_size=0.20
seed=7
X_train, X_validation, y_train, y_validation=model_selection.train_test_split(X,y,test_size=val_size,random_state=seed)

clf=LogisticRegression()
clf.fit(X_train,y_train)
print (clf.score(X_train,y_train))
print (clf.score(X_validation,y_validation))

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
print (knn.score(X_validation,y_validation))

predictions=knn.predict(X_validation)
acc=accuracy_score(y_validation,predictions)
print(acc)


print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))