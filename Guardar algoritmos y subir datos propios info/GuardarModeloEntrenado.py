# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 22:55:39 2020

@author: alexa
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#la libreria que usamos para guardar el algoritmo es joblib
from sklearn.externals import joblib


iris = load_iris()

clf = KNeighborsClassifier(n_neighbors = 5)

X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target)

clf.fit(X_train,y_train)

score = clf.score(X_test,y_test)

print(score)

# De esta forma guardamos el algoritmo en formato pkl (una especie de ejecutable de python en otros programas)
joblib.dump(clf, 'algoritmoIrisEntrenado.pkl')

