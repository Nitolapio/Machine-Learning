# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:30:29 2020

@author: alexa
"""

import sklearn
import mglearn
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  #Este es el modelo que vamos a usar

mglearn.plots.plot_pca_illustration()   #Ejemplo de grafica PCA

cancer = load_breast_cancer()
#Este set de datos cuenta con 30 características. No podemos graficar 30 características;
# ya que necesitariamos 30 dimensiones. Por eso usamos PCA: graficamos en 2D todas las características
# y eliminamos todo el ruido de los datos

pca = PCA(n_components = 2) #Los componentes son los ejes de los gráficos como podemos ver
                            # en el ejemplo generado en la primera línea de código

pca.fit(cancer.data)

data_transformada = pca.transform(cancer.data)

print(cancer.data.shape)  #Esto cuenta con 30 dimensiones
print(data_transformada.shape)    #Vemos que ahora toda la data se ha transformado en una representacion de 2D

mglearn.discrete_scatter(data_transformada[:,0], data_transformada[:, 1], cancer.target)
plt.legend(cancer.target_names, loc='best')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


from sklearn.preprocessing import MinMaxScaler  #vamos a usar un rango binario (1-0) para polarizar los datos
escala = MinMaxScaler()
escala.fit(cancer.data)
escalada = escala.transform(cancer.data) 
pca.fit(escalada)
transformada = pca.transform(escalada)
mglearn.discrete_scatter(transformada[:,0], transformada[:,1], cancer.target)
plt.legend(cancer.target_names, loc = 'best')
plt.gca()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

print(escalada)

print(escalada.data)




