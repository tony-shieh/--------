# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:36:14 2021

@author: Class
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model  import LinearRegression

train_data = pd.read_csv('insurance.csv') 
X_data = train_data.drop(['charges'], axis = 1)
y_data = train_data['charges'] 


X_data = X_data.dropna(axis = 1, how = 'any')


encoder = LabelEncoder()
X_encoded = pd.DataFrame(X_data,columns=X_data.columns).apply(lambda col:encoder.fit_transform(col))
kbest = SelectKBest(f_regression,k = 6)
X_new = kbest.fit_transform(X_encoded, y_data)


X_train, X_test, y_train, y_test = train_test_split(X_new, y_data,train_size=0.7, test_size=0.3,random_state=0)
model = LinearRegression()

model.fit(X_train, y_train)
score =model.score(X_test, y_test)
print('Score: ', score)
print('Accuracy: ' + str(score*100) + '%')


