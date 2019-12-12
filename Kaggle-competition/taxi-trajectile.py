#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:44:13 2019

@author: utkarsh
"""

import pandas as pd
import numpy as np
import seaborn as sns

train_df = pd.read_csv('../dataSet/pkdd-15-predict-taxi-service-trajectory-i/train.csv')
test_df = pd.read_csv('../dataSet/pkdd-15-predict-taxi-service-trajectory-i/test.csv')

polyline = train_df['POLYLINE']

null_columns = train_df.columns[train_df.isnull().any()]
null_column1=train_df[null_columns1].isnull().sum()

train_df['ORIGIN_CALL']=train_df['ORIGIN_CALL'].fillna(train_df['ORIGIN_CALL'].mean())
train_df['ORIGIN_STAND']=train_df['ORIGIN_STAND'].fillna(train_df['ORIGIN_STAND'].mode()[0])


value_count_data=train_df['CALL_TYPE'].value_counts()

columns = ['CALL_TYPE','DAY_TYPE','MISSING_DATA']

final_df=pd.concat([train_df,test_df],axis=0)

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

final_df=category_onehot_multcols(columns)

a = np.array(final_df['POLYLINE'])
meanvalue = np.mean(a,axis=1)
    