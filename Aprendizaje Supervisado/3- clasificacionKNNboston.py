# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:29:59 2020

@author: alexa
"""

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

type(cancer)

cancer.keys()


X_train,X_test,y_train,y_test = train_test_split(cancer['data'], cancer['target'])

print(X_train.shape)
print(y_train.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)

print(knn.fit(X_test, y_test))

print(knn.score(X_test, y_test))


