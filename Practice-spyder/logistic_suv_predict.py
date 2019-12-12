#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:29:38 2019

@author: utkarsh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the datasets

datasets = pd.read_csv('social_network_adv.csv')

#print(datasets.isnull().sum())

X = datasets.iloc[:,[2,3]].values
y= datasets.iloc[:,4].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# read
from sklearn.preprocessing import StandardScaler
#scale down input value for better performance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_pred)
