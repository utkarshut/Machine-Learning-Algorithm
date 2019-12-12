#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:35:30 2019

@author: utkarsh
"""

import pandas as pd
#for numerical computaions we can use numpy library
import numpy as np

datasets = pd.read_csv("./dataSet/iris/iris.csv")
# Print the first 5 rows of the dataframe.
x = datasets.iloc[:,1:5]
y = datasets.iloc[:,5]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

#metric model to check accuracy
from sklearn import metrics
#try running k = 1 to k 25 record testing accuracy
k_range = range(1,26)
scores={}
scores_list =[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors =k)
    knn.fit(x_train,y_train)
    y_pred =knn.predict(x_test)
    scores[k]=metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))
import matplotlib.pyplot as plt

#graph between k and accuracy

plt.plot(k_range,scores_list)
plt.xlabel('value of K in KNN')
plt.ylabel('Testing accuracy')

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y)

KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',
                     metric_params=None,n_jobs=1,n_neighbors=5,p=2,weights='uniform')

classes = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginca'}

#making prediction on unseen data

x_new =[[3,4,5,2],[5,4,2,2]]
y_predict =knn.predict(x_new)

print(y_predict[0])
print(y_predict[1])


