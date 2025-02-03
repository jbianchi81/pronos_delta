import json
import pandas as pd

# Cargar el JSON desde un archivo
with open("Estaciones.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Transformar el JSON en una lista de diccionarios
rows = []
for key, value in data.items():
    row = {"id": key}  # Agregar el identificador de la estación
    row.update(value)   # Agregar el resto de los datos
    rows.append(row)

# Crear un DataFrame y guardarlo en un CSV
df = pd.DataFrame(rows)


df.to_csv("Estaciones.csv", index=False, encoding="ISO-8859-1")

print("Archivo CSV guardado exitosamente.")