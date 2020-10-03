#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:47:39 2020

@author: nitolapio
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



fashion_mnist = keras.datasets.fashion_mnist # Con esto importamos el dataset de Fashion_MNIST

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #Cargamos los datos y los separamos en entrenamiento y test
#No puedo seguir porque no se carga bien el data en test_images

"""
Las imagenes son 28x28 arreglos de NumPy, con valores de pixel que varian de 0 a 255. Los labels son un arreglo de 
integros, que van del 0 al 9. Estos corresponden a la class de ropa que la imagen representa.
"""

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Es conveniente probar y ver los datos que tenemos antes de usarlos haciendo print o mirándolo en el explorador de variables


# De esta manera graficamos la imagen
"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
"""

# Es importante re-escalar los valores de 255 a intervalos de 0 a 1 para analizarse mejor
train_images = train_images / 255.0
test_images = train_images / 255.0

#Verificamos el formato con el siguiente gráfico
"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary) 
    plt.xlabel(class_names[train_labels[i]])
    plt.show()
"""

# Aquí creamos el modelo que vamos a usar. En este caso, el Sequential
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),    # Ponemos toda la matriz de píxeles bidimensionales en una hilera unidimensional 28*28 = 784 datos de entrada
    keras.layers.Dense(128, activation='relu'),    # Capas densamente conectadas con 128 nodos (neuronas)
    keras.layers.Dense(10, activation = 'softmax'),  # 10 nodos que devuelve un array de 10 probabilidades que llegan hasta 1 según el peso que le llegue.
    ])

#Añadimos unos parámetros más y compilamos
model.compile(
    optimizer = 'adam',     # Sirve para ver cómo el modelo avanza (bien o mal)
    loss='sparse_categorical_crossentropy',  ## Mide exactitud en un momento dado. Se quiere minimizar esta función para mejorar el modelo a medida avanza
    metrics=['accuracy']) # Sirve para monitorear los pasos de entrenamiento y de pruebas)


#Entrenamos el modelo
model.fit(train_images, train_labels, epochs=10)   # Tenemos una exactitud del 91½, lo cual es perfecto

#Ahora tenemos que evaluar la exactitud de los resultados del modelo
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)   #Verbose sirve para poner 3 modos de enseñar los datos como queramos



































