import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

#cargamos las 4 combinaciones de las compuertas XOR
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

#estos son los resultados que se obtienen en el mismo orden
target_data = np.array([[0],[1],[1],[0]], "float32")


#Arquitectura de la red neuronal
model = Sequential()
model.add(Dense(16, input_dim = 2, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


#Ajustes del modelo
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['binary_accuracy'])

#entrenamiento de la red
model.fit(training_data, target_data, epochs = 1000)






#Evaluamos y predecimos
scores = model.evaluate(training_data, target_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#predicciones posibles
print (model.predict(training_data).round())
