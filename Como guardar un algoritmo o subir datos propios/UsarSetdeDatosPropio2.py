#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:05:25 2020

@author: nitolapio
"""

import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Modelo de regresión
reg = linear_model.LogisticRegression()

archivo = "irisdatos.csv"

df = pd.read_csv(archivo)   # En la variable df ya tenemos el set de datos guardado

# Separamos filas y columnas
arrayx = df[df.columns[:-1]].as_matrix()
arrayy = df[df.columns[-1].as_matrix()]

X_train, X_test, y_train, y_test = train_test_split(arrayx, arrayy)

reg.fit(X_train, y_train)

reg.score(X_test, y_test)
