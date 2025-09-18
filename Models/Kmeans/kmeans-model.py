import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================
# Cargar datos
# ==========================
df = pd.read_csv("../../Data/Processed_Activities.csv")

# Variables numéricas (sin incluir la etiqueta)
X = df.drop(columns=["tipo_de_actividad"])

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# Visualización antes de clasificar
# ==========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, cmap="viridis")
plt.title("Datos Originales (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# ==========================
# Método del codo
# ==========================
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("WCSS")
plt.title("Método del codo - KMeans")
plt.show()

# ==========================
# Entrenar con k clusters
# ==========================
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Obtener los centroides
centroids = kmeans.cluster_centers_

# Transformar los centroides al espacio PCA
centroids_pca = pca.transform(centroids)

# ==========================
# Visualización después de clasificar
# ==========================
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap="viridis", alpha=0.6, label="Datos")
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="red", marker="X", s=200, label="Centroides")
plt.title("Clusters y Centroides (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.legend()
plt.grid(True)
plt.show()

# Conteo por grupo
print(df["cluster"].value_counts())
