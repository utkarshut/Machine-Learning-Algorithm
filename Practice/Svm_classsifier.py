#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:23:36 2019

@author: utkarsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bankdata = pd.read_csv("./dataSet/bill_authentication.csv")


bankdata.shape

bankdata.head()

X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

