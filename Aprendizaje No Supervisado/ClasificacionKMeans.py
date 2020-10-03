# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:30:41 2020

@author: alexa
"""


from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn import metrics  #para ver qué bien ha aprendido el algoritmo

iris = load_iris()

X = iris.data
y = iris.target

km = KMeans(n_clusters=3,max_iter=3000)
#n_clusters determina el numero de grupos en los que queremos clasificar nuestros datos
#max_iter es la cantidad de iteraciones que hace el algoritmo para llegar a clasificar correctamente

km.fit(X)

predicciones = km.predict(X) 

score = metrics.adjusted_rand_score(y, predicciones)
print(score)