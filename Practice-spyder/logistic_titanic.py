#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:09:40 2019

@author: utkarsh
"""

import pandas as pd
#for numerical computaions we can use numpy library
import numpy as np
#stastical bondig
import seaborn as sns
import matplotlib.pyplot as plt
import math

titanic = pd.read_csv("./dataSet/titanic/train.csv")
#print(titanic.shape)
#print(titanic.describe)
#sns.countplot(x='Survived',data=titanic)
#sns.countplot(x='Survived',hue='Sex', data=titanic)
#sns.countplot(x='Survived',hue='Pclass', data=titanic)

#titanic['Age'].plot.hist()

#titanic['Fare'].plot.hist(bins=20,figsize=(10,5))

# titanic.info()


#sns.countplot(x='SibSp', data=titanic)

#print(titanic.isnull())
#print(titanic.isnull().sum())
#sns.heatmap(titanic.isnull(),yticklabels=False,cmap='viridis')

#sns.boxplot(x='Pclass',y='Age',data=titanic)
titanic.drop('Cabin',axis=1,inplace=True)

titanic.dropna(inplace=True)
#print(titanic.isnull().sum())
sex = pd.get_dummies(titanic['Sex'],drop_first=True)
embark = pd.get_dummies(titanic['Embarked'],drop_first=True)
pclass = pd.get_dummies(titanic['Pclass'],drop_first=True)

#concat

titanic = pd.concat([titanic,sex,embark,pclass],axis=1)

#drop unneccesary

titanic.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1,inplace=True)
#train data

X =titanic.drop('Survived',axis=1)
#predicting
y=titanic['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

#accuracy check
from sklearn.metrics import classification_report

report=classification_report(y_test,predictions)
#accuracy

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predictions)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)


