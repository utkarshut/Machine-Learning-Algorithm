#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:34:48 2019

@author: utkarsh
"""
# We can use the pandas library in python to read in the csv file.
import pandas as pd
#for numerical computaions we can use numpy library
import numpy as np

titanic = pd.read_csv("./dataSet/titanic/train.csv")
# Print the first 5 rows of the dataframe.
t1 = titanic.head()

titanic_test = pd.read_csv("./dataSet/titanic/test.csv")
t2 = titanic_test.head()
#transpose
t3 =titanic_test.head().T
#note their is no Survived column here which is our target varible we are trying to predict

#shape command will give number of rows/samples/examples and number of columns/features/predictors in dataset
#(rows,columns)
titanic.shape
#Describe gives statistical information about numerical columns in the dataset
titanic.describe()
#you can check from count if there are missing vales in columns, here age has got missing values

#info method provides information about dataset like 
#total values in each column, null/not null, datatype, memory occupied etc
titanic.info()

#lets see if there are any more columns with missing values 

null_columns = titanic.columns[titanic.isnull().any()]
print("check:",titanic.isnull())
print("check:",titanic.isnull().any())
print("column values:",null_columns)
print(titanic.isnull().sum())

#Age, Fare and cabin has missing values.
#we will see how to fill missing values next.**

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)

plt.style.use('ggplot')
labels = []
values = []
for col in null_columns:
    labels.append(col)
    values.append(titanic[col].isnull().sum())
    
ind = np.arange(len(labels))
width=0.9
fig, ax = plt.subplots(figsize=(6,5))
rects = ax.barh(ind, np.array(values), color='purple')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Variables with missing values");


titanic.hist(bins=10,figsize=(7,7),grid=False);

#**we can see that Age and Fare are measured on very different scaling. So we need to do feature scaling before predictions.**

g = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="purple");

g = sns.FacetGrid(titanic, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();

g = sns.FacetGrid(titanic, hue="Survived", col="Sex", margin_titles=True,
                palette="Set1",hue_kws=dict(marker=["^", "v"]))
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)
g.fig.suptitle('Survival by Gender , Age and Fare');

titanic.Embarked.value_counts().plot(kind='bar', alpha=0.55)
plt.title("Passengers per boarding location");

sns.factorplot(x = 'Embarked',y="Survived", data = titanic,color="r");

sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=titanic, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many Men and Women Survived by Passenger Class');



