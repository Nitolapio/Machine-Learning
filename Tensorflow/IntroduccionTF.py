#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:02:44 2020

@author: nitolapio
"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # Improtante para que no salga error con placeholder

constante = tf.constant([2.0,3,4], dtype = tf.float32, name='Constante1') # Vector con 3 elementos

apartado = tf.compat.v1.placeholder(tf.float32, name= 'Apartado1') # Placeholder es una variable que vamos cambiando a medida ejecutamos el programa


variable = tf.compat.v1.Variable(3, dtype=tf.float32, name='Variable1') #Tipo de dato que cambia de valor también

matriz = tf.zeros([3,4], tf.int32, name='Matriz1') # Creamos una matriz vacía que sirve de mucho

# Hasta aquí hemos creado nuestra gráfica computacional



# Aquí empezamos nuestro programa

inicializar = tf.compat.v1.global_variables_initializer() #Inicializador
sess = tf.compat.v1.Session() # Esta es la sesión sobre la cual se inicializa todo el gráfico

sess.run(inicializar)  # Con esto inicializamos la sesión

print(sess.run(matriz))  # Ejemplo de print

mutliplicacion = apartado * constante


