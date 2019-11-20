#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:43:04 2019

@author: utkarsh
"""
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
 

 
dataset = datasets.load_iris()

model = GaussianNB()
model.fit(dataset.data, dataset.target)


expected = dataset.target
predicted = model.predict(dataset.data)
print("SDF")