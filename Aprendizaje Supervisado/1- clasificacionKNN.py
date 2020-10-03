# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:05:22 2020

@author: alexa
"""

import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

type(iris)

iris.keys()
iris['data']
iris['target_names']
iris['target']
iris['feature_names']


X_train,X_test,y_train,y_test = train_test_split(iris['data'], iris['target'])

print(X_train.shape) 
print(y_train.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)

print(knn.fit(X_test,y_test))

print(knn.score(X_test,y_test))

print(knn.predict([[1.2,3.2,5.6,1.1]]))

