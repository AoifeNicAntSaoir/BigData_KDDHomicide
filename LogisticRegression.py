#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#loading dataset and separating predictors from predicted variable
df = pd.read_csv('database.csv', low_memory=False)

df = df[(df['Crime Solved']=='Yes') & (df['Perpetrator Race']!='Unknown') & (df['Victim Sex']!='Unknown')]

X = df[['Year', 'Victim Ethnicity', 'Victim Sex', 'Victim Age', 'Victim Race']]

X = pd.get_dummies(X)

Y = df['Perpetrator Sex']
#Y = pd.get_dummies(Y)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)


lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(predictions)
print(predictions[0:5])
print('Score:', lr.score(X_test, y_test))

