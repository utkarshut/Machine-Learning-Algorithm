a#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 23:39:53 2019

@author: utkarsh
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Load in the data
dataset = pd.read_csv('./dataSet/mushrooms.csv')

#Define the column names
dataset.columns = ['target','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population',
'habitat']
for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])
 
#Display information about the data set
print(dataset.info())
 
X = dataset.drop(['target'], axis=1)
Y = dataset['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = DecisionTreeClassifier(criterion='entropy', max_depth=1)
AdaBoost = AdaBoostClassifier(base_estimator=model, n_estimators=400, learning_rate=1)

#Fit the model with training data
boostmodel = AdaBoost.fit(X_train, Y_train)

#Evaluate the accuracy of the model
y_pred = boostmodel.predict(X_test)
predictions = metrics.accuracy_score(Y_test, y_pred)
#Calculating the accuracy in percentage
print('The accuracy is: ', predictions * 100, '%')