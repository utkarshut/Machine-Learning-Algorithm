#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:21:19 2019

@author: utkarsh
"""

# -*- coding: utf-8 -*-

# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

train_df = pd.read_csv('../dataSet/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../dataSet/house-prices-advanced-regression-techniques/test.csv')
combine = [train_df, test_df]
describe=train_df.describe()

train_df.info()
valueCountData=train_df['MSZoning'].value_counts()
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False)

## Fill Missing Values

train_df['LotFrontage']=train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean())
train_df.drop(['Alley'],axis=1,inplace=True)

train_df['BsmtCond']=train_df['BsmtCond'].fillna(train_df['BsmtCond'].mode()[0])
train_df['BsmtQual']=train_df['BsmtQual'].fillna(train_df['BsmtQual'].mode()[0])

train_df['FireplaceQu']=train_df['FireplaceQu'].fillna(train_df['FireplaceQu'].mode()[0])
train_df['GarageType']=train_df['GarageType'].fillna(train_df['GarageType'].mode()[0])

train_df.drop(['GarageYrBlt'],axis=1,inplace=True)

train_df['GarageFinish']=train_df['GarageFinish'].fillna(train_df['GarageFinish'].mode()[0])
train_df['GarageQual']=train_df['GarageQual'].fillna(train_df['GarageQual'].mode()[0])
train_df['GarageCond']=train_df['GarageCond'].fillna(train_df['GarageCond'].mode()[0])

train_df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

train_df.shape

train_df.drop(['Id'],axis=1,inplace=True)

train_df['MasVnrType']=train_df['MasVnrType'].fillna(train_df['MasVnrType'].mode()[0])
train_df['MasVnrArea']=train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].mode()[0])
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')

train_df['BsmtExposure']=train_df['BsmtExposure'].fillna(train_df['BsmtExposure'].mode()[0])

train_df['BsmtFinType2']=train_df['BsmtFinType2'].fillna(train_df['BsmtFinType2'].mode()[0])

train_df.dropna(inplace=True)

null_columns1 = train_df.columns[train_df.isnull().any()]
null_column1=train_df[null_columns1].isnull().sum()

##HAndle Categorical Features

columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

len(columns)

def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


main_df=train_df.copy()

## Combine Test Data 

test_df=pd.read_csv('../dataSet/house-prices-advanced-regression-techniques/formulatedtest.csv')

final_df=pd.concat([train_df,test_df],axis=0)

final_df=category_onehot_multcols(columns)

final_df =final_df.loc[:,~final_df.columns.duplicated()]


df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]

#df_Test.drop(['SalePrice'],axis=1,inplace=True)

y_train=df_Train['SalePrice']
X_train=df_Train.drop(['SalePrice'],axis=1)

from sklearn.ensemble import RandomForestClassifier

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(df_Test.drop(['SalePrice'],axis=1))
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

pred = pd.DataFrame(Y_pred)
sub_df = pd.read_csv('../dataSet/house-prices-advanced-regression-techniques/sample_submission.csv') 
datasets = pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('../dataSet/house-prices-advanced-regression-techniques/submission.csv',index=False)


