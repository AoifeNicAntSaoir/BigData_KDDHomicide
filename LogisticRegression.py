# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 22:19:38 2018

@author: Aoife Sayers

"""
#Tutorial link: https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('database.csv', low_memory=False)

df = df[(df['Crime Solved']=='Yes') & (df['Perpetrator Race']!='Unknown')]
df = df[['Perpetrator Sex','Perpetrator Race']]

y=df['Perpetrator Sex']
df = pd.get_dummies(df, columns=['Perpetrator Sex','Perpetrator Race'])

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)


lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(predictions)
print(predictions[0:5])