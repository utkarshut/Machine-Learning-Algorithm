# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

dataSet = pd.read_csv('salaryexp.csv')

x = dataSet.iloc[:,:1].values
y = dataSet.iloc[:,1:2].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

simpleLinearRegression = LinearRegression()

simpleLinearRegression.fit(x_train,y_train)

predct = simpleLinearRegression.predict(x_test)

predcts = simpleLinearRegression.predict(x_train)

plt.scatter(x_train, y_train, color = 'red')

plt.plot(x_train, simpleLinearRegression.predict(x_train), color = 'blue')

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, simpleLinearRegression.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
