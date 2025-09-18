import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, cross_val_predict

# ==============================
# 1. Cargar datos
# ==============================
df = pd.read_csv(r"Data\Processed_Activities.csv")

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

# KFold Cross Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# PCA for dimensionality reduction (retain 95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced the feature space from {X_scaled.shape[1]} to {X_pca.shape[1]} components.")

# Logistic Regression Model with PCA features and class_weight balanced
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
cv_scores_logreg = cross_val_score(model, X_pca, y_encoded, cv=kf, scoring='accuracy')
print("\nKFold Cross-Validation Scores (Logistic Regression with PCA, balanced):", cv_scores_logreg)
print("Mean CV Accuracy (Logistic Regression with PCA, balanced):", cv_scores_logreg.mean())

# Cross-validated ROC AUC (macro average)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_proba_cv = cross_val_predict(model, X_pca, y_encoded, cv=skf, method='predict_proba')

# For binary classification, use the positive class probabilities directly
if len(le.classes_) == 2:
    # Binary case: use probabilities for the positive class (class 1)
    roc_auc_cv = roc_auc_score(y_encoded, y_pred_proba_cv[:, 1])
    print(f"Cross-validated ROC AUC Score (binary): {roc_auc_cv:.4f}")
    
    # Plot ROC curve for binary classification
    fpr, tpr, _ = roc_curve(y_encoded, y_pred_proba_cv[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cross-Validated ROC AUC Curve (Binary Classification)')
    plt.legend(loc="lower right")
    plt.show()
else:
    # Multiclass case
    y_binarized_cv = label_binarize(y_encoded, classes=le.transform(le.classes_))
    roc_auc_cv = roc_auc_score(y_binarized_cv, y_pred_proba_cv, average="macro")
    print(f"Cross-validated ROC AUC Score (macro average): {roc_auc_cv:.4f}")

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', len(le.classes_))
    for i, class_name in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_binarized_cv[:, i], y_pred_proba_cv[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors(i), lw=2,
                 label=f'ROC curve of class {class_name} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cross-Validated ROC AUC Curve for Multiclass Logistic Regression')
    plt.legend(loc="lower right")
    plt.show()

# Train/test split on PCA features
X_train_pca, X_test_pca, _, _ = train_test_split(
    X_pca, y_encoded, test_size=0.17, random_state=42, stratify=y_encoded
)
model.fit(X_train_pca, y_train)


# ==============================
# 6. Evaluación
# ==============================
y_pred = model.predict(X_test_pca)

# Obtener las clases presentes en y_test
present_classes = unique_labels(y_test)

# Generar el reporte de clasificación con las clases presentes
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_[present_classes]))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=present_classes)
ConfusionMatrixDisplay(cm, display_labels=le.classes_[present_classes]).plot(cmap="Blues")
plt.show()

# ==============================
# 7. ROC AUC Curve (Multiclass)
# ==============================




# Binarize the output for multiclass ROC AUC
y_test_binarized = label_binarize(y_test, classes=present_classes)
y_score = model.decision_function(X_test_pca)
if y_score.ndim == 1:
    y_score = y_score.reshape(-1, 1)

if y_test_binarized.shape[1] > 1:
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_test_binarized.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', y_test_binarized.shape[1])
    for i, class_idx in enumerate(present_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                 label=f'ROC curve of class {le.classes_[class_idx]} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve for Multiclass Logistic Regression')
    plt.legend(loc="lower right")
    plt.show()
else:
    print("\nROC AUC curve not plotted: Only one class present in y_test.")

# ==============================
# 8. ROC AUC Score (Multiclass)
# ==============================
# Calculate and print the overall ROC AUC score (macro average)
roc_auc_score_macro = roc_auc_score(y_test_binarized, y_score, average="macro")
print(f"\nROC AUC Score (macro average): {roc_auc_score_macro:.4f}")
