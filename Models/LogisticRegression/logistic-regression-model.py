import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels

# ==============================
# 1. Cargar datos
# ==============================
df = pd.read_csv("../../Data/Processed_Activities.csv")

# ==============================
# 2. Conocer los labels
# ==============================
# Obtener los labels únicos
labels = df["tipo_de_actividad"].unique()
print("Labels únicos en los datos:", labels)

# Contar el número de datos por label
label_counts = df["tipo_de_actividad"].value_counts()
print("\nNúmero de datos por label:")
print(label_counts)

# ==============================
# 3. Graficar la distribución de los labels
# ==============================
plt.figure(figsize=(8, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
plt.title("Distribución de los datos por tipo de actividad")
plt.xlabel("Tipo de Actividad")
plt.ylabel("Número de Datos")
plt.xticks(rotation=45)
plt.show()

# Variables predictoras y objetivo
X = df[["distancia", "calorias", "tiempo",
        "frecuencia_cardiaca_media", "fc_maxima", "tiempo_en_movimiento"]]
y = df["tipo_de_actividad"]

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 4. Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.17, random_state=42, stratify=y_encoded
)

# ==============================
# 5. Modelo
# ==============================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ==============================
# 6. Evaluación
# ==============================
y_pred = model.predict(X_test)

# Obtener las clases presentes en y_test
present_classes = unique_labels(y_test)

# Generar el reporte de clasificación con las clases presentes
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_[present_classes]))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=present_classes)
ConfusionMatrixDisplay(cm, display_labels=le.classes_[present_classes]).plot(cmap="Blues")
plt.show()
