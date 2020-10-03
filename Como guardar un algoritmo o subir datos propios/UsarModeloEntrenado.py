# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:08:41 2020

@author: alexa
"""

from sklearn.datasets import load_iris
from sklearn.externals import joblib

iris = load_iris()

# de esta forma importamos el algoritmo
clf = joblib.load('algoritmoIrisEntrenado.pkl')

score = clf.score(iris.data, iris.target)

print(score)
