# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:58:06 2020

@author: alexa
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   #Sirve para graficar en 3D
from matplotlib import cm    # Colormap
plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("./articulos_ml.csv")

descripcion = data.describe()

data.drop(['Title', 'url', 'Elapsed days'], 1).hist()
plt.show()  #Valores en los que se concentran la mayoría de registros
