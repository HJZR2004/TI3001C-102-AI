import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.io as pio
pio.renderers.default = 'browser'


# Leer el archivo CSV
data_frame = pd.read_csv('../../Data/Processed_Activities.csv')

# Preparar los datos
X_scaled = data_frame[['distancia', 'calorias', 'tiempo', 'frecuencia_cardiaca_media', 'fc_maxima']]

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaled)

# K-Means en train (K=n)
kmeans = KMeans(n_clusters=4, random_state=42, init='k-means++', n_init=10, max_iter=300)
clusters = kmeans.fit_predict(X_scaled)

# Agregar los clusters al DataFrame
data_frame['Cluster'] = clusters

print(data_frame.head())


# Visualización en 2D con Plotly Express
fig = px.scatter(
    data_frame,
    x='distancia',  # Eje X
    y='calorias',   # Eje Y
    color='Cluster',  # Colorear por cluster
    title='Clusters visualizados en 2D',
    labels={'Cluster': 'Cluster'}
)

# Mostrar el gráfico
fig.show()

# Visualización en 3D con Plotly Express
fig = px.scatter_3d(
    data_frame,
    x='distancia',  # Eje X
    y='calorias',   # Eje Y
    z='tiempo',     # Eje Z
    color='Cluster',  # Colorear por cluster
    title='Clusters visualizados en 3D',
    labels={'Cluster': 'Cluster'}
)

# Mostrar el gráfico
fig.show()