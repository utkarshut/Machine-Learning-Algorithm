#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:39:54 2019

@author: utkarsh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the datasets

datasets = pd.read_csv('social_network_adv.csv')
x = datasets.iloc[:,2:4]
y = datasets.iloc[:,4]

 

#feature scalinh
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


from sklearn.svm import SVC

classifier = SVC(kernel='linear',random_state=0)

classifier.fit(x_train,y_train)


y_pred =classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)