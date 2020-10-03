# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:55:25 2020

@author: alexa
"""
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

boston = load_boston()

print(boston.data.shape)
print(boston.target.shape)

X_train,X_test,y_train,y_test = train_test_split(boston.data, boston.target)

print(X_train.shape)
print(y_test.shape)

knn = KNeighborsRegressor(n_neighbors = 7)

knn.fit(X_train,y_train)

print(knn.score(X_test,y_test))

del knn  #Borramos KNN
rl = LinearRegression()

rl.fit(X_train,y_train)

print(rl.score(X_test,y_test))

del rl

ridge = Ridge(alpha = 0.5)

ridge.fit(X_train,y_train)
print(ridge.score(X_test,y_test))


