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


import xgboost
classifier=xgboost.XGBRegressor()

import xgboost
regressor=xgboost.XGBRegressor()

booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]

## Hyper Parameter Optimization


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)

random_cv.fit(X_train,y_train)

random_cv.best_estimator_

regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)


regressor.fit(X_train,y_train)


import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


y_pred=regressor.predict(df_Test.drop(['SalePrice'],axis=1))




