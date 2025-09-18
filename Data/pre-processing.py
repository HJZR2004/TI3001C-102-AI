import pandas as pd
import numpy as np

# Leer el archivo CSV original
df = pd.read_csv('Activities.csv')

# Mostrar información inicial del DataFrame
# print(df.info())

# Seleccionar las columnas relevantes y renombrarlas
df = df[['Tipo de actividad', 'Distancia', 'Calorías', 'Tiempo', 'Frecuencia cardiaca media', 'FC máxima', 'Tiempo en movimiento']]
df.columns = [n.lower().replace(' ', '_').replace('á', 'a').replace('í', 'i') for n in df.columns]

# Limpiar la columna 'calorias' para eliminar comas y convertir a entero
df['calorias'] = df['calorias'].str.replace(',', '').astype(int)

# Convertir la columna 'tiempo' a minutos
df['tiempo'] = df['tiempo'].apply(lambda x: round(int(x.split(':')[0]) * 60 + int(x.split(':')[1]) + float(x.split(':')[2]) / 60, 2))

# Convertir la columna 'tiempo_en_movimiento' a minutos
df['tiempo_en_movimiento'] = df['tiempo_en_movimiento'].apply(lambda x: round(int(x.split(':')[0]) * 60 + int(x.split(':')[1]) + float(x.split(':')[2]) / 60, 2))

# Guardar el DataFrame procesado en un nuevo archivo CSV
df.to_csv('Processed_Activities.csv', index=False)