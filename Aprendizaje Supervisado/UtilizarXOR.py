
# serializar el modelo a JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serializar los pesos a HDF5
mode.save_weights("model.h5")
print("Modelo Guardado!")


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# cargar pesos al nuevo modelo
loaded_model.load_weights("model.h5")
print("Cargado modelo desde disco.")

#compilar modelo cargado y listo para usar
loaded_model.compile(loss='mean_squared_error', optimizer = 'adam', metrics= ['binary_accuracy'])
