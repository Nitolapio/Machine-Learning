# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:07:07 2020

@author: alexa
"""

"""" Mediante los algoritmos de smv podremos ver cuánto de seguro está nuestro
sistema de predicción de las clasificaciones que hace. con decision_function vemos cuán separado está
un dato de una línea separadora creado por el algoritmo. Con predict_proba es lo mismo pero se muestra con probabilidades,
y con predict, directamente predice los resultados.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm    #Vectores de soporte: traza hiperplano para clasificar data

iris = load_iris()

X_train,X_test,y_train,y_test = train_test_split(iris.data, iris.target)

algoritmo = svm.SVC(probability = True)

algoritmo.fit(X_train,y_train)

algoritmo.decision_function_shape = "ovr"

decision_function = algoritmo.decision_function(X_test)[:10]  #Número de decisiones 10

predict_proba = algoritmo.predict_proba(X_test)[:10]

predict = algoritmo.predict(X_test)[:10]