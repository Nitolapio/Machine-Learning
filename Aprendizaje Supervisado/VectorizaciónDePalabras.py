# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:13:43 2020

@author: alexa
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#mirar libreria nlkt

noticias = fetch_20newsgroups(subset = 'train')

vector = CountVectorizer()

vector.fit(noticias.data)

vector.vocabulary_  #devuelve la forma en la que ha vectorizado las palabras del dataset

bolsa = vector.transform(noticias.data)  #Convertimos los datos en una matriz

bolsay = noticias.target

X_train,X_test,y_train,y_test = train_test_split(bolsa, bolsay)

lr = LogisticRegression()

lr.fit(X_train,y_train)

print(lr.score(X_test,y_test))