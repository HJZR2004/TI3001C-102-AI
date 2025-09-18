# =============================================================
# Proyecto: Tipos de entrenamiento (Garmin) con K-Means + Reg. Logística
# - Descubrimiento de grupos (no supervisado) con K-Means
# - Clasificación supervisada (pseudo-etiquetas de K-Means) con Regresión Logística
# - Split 83/17 (como 100/20 de 120), SIN data leakage
# - Métricas: accuracy, precision, recall, F1, ROC/AUC (multiclase)
# - Gráficas simples: codo (WCSS), PCA 2D, barras por grupo, boxplots
# - Inferencia demo sin helpers
# =============================================================

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support
)

# Utilidades
# -----------------------------
def to_float_es(s):
    """Convierte '1,057' o '--' a float (np.nan si vacío)."""
    if s is None: return np.nan
    s = str(s).strip()
    if s in ("", "--"): return np.nan
    s = s.replace(",", "")
    try: return float(s)
    except ValueError: return np.nan

def tiempo_a_minutos(s):
    """Convierte 'HH:MM:SS' o 'MM:SS' a minutos (float)."""
    if s is None: return np.nan
    s = str(s).strip()
    if s in ("", "--"): return np.nan
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
            return h*60.0 + m + sec/60.0
        elif len(parts) == 2:
            m = float(parts[0]); sec = float(parts[1])
            return m + sec/60.0
        else:
            return float(s)/60.0
    except ValueError:
        return np.nan

def find_col(headers, target_prefix):
    """Busca una columna cuyo nombre comience con target_prefix (insensible a mayúsculas)."""
    for h in headers:
        if h.lower().startswith(target_prefix.lower()):
            return h
    return None

# 1) Cargar CSV y derivar features
# -----------------------------
csv_path = Path(r"C:\Users\melis\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Tec\Semestre 7\ProyectoFinalB1\Activities.csv")
rows = []
with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = [h.strip() for h in reader.fieldnames]
    col_dist = find_col(headers, "Distancia")
    col_time = find_col(headers, "Tiempo")
    col_time_mov = find_col(headers, "Tiempo en movimiento")
    col_fc = find_col(headers, "Frecuencia cardiaca media")
    col_cal = find_col(headers, "Calorías")

    for row in reader:
        dist = to_float_es(row[col_dist]) if col_dist else np.nan
        tmin = tiempo_a_minutos(row[col_time_mov]) if (col_time_mov and row.get(col_time_mov)) else (
               tiempo_a_minutos(row[col_time]) if (col_time and row.get(col_time)) else np.nan)
        fc   = to_float_es(row[col_fc]) if col_fc else np.nan
        cal  = to_float_es(row[col_cal]) if col_cal else np.nan
        pace = (tmin / dist) if (dist and dist > 0 and tmin and tmin > 0) else np.nan
        rows.append((dist, tmin, pace, fc, cal))

df = pd.DataFrame(rows, columns=["Distancia_km","Tiempo_min","Pace_min_km","FC_media_bpm","Calorias_kcal"])
df = df.dropna().reset_index(drop=True)

# 2) Filtro de outliers
# -----------------------------
mask = (
    (df["Distancia_km"] > 0.5) & (df["Distancia_km"] <= 60) &
    (df["Tiempo_min"] >= 5) & (df["Tiempo_min"] <= 400) &
    (df["Pace_min_km"] >= 3) & (df["Pace_min_km"] <= 12) &
    (df["FC_media_bpm"] >= 80) & (df["FC_media_bpm"] <= 210) &
    (df["Calorias_kcal"] >= 50) & (df["Calorias_kcal"] <= 3000)
)
df_clean = df.loc[mask].reset_index(drop=False).rename(columns={"index":"row_idx_original"})

# 3) Matriz de features + escalado (para exploración no supervisada)
# -----------------------------
features = ["Distancia_km","Tiempo_min","Pace_min_km","FC_media_bpm","Calorias_kcal"]
X = df_clean[features].values
if X.shape[0] < 120:
    raise SystemExit(f"Se requieren >=120 ejemplos tras limpieza; quedaron {X.shape[0]}. Ajusta umbrales del paso 2.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Elegir K con método del codo (solo WCSS/inercia) - Exploración
# -----------------------------
Ks = [2,3,4,5,6]
inertias, models = [], []

for k in Ks:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)   # = WCSS
    models.append(km)
    print(f"[K={k}] inercia (WCSS) = {km.inertia_:.1f}")

plt.figure()
plt.plot(Ks, inertias, marker='o')
plt.title("Método del codo (Inercia/WCSS vs K)")
plt.xlabel("K"); plt.ylabel("Inercia (WCSS)"); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("elbow_kmeans.png", dpi=150)

# Para el proyecto queremos 3 tipos -> fijamos K=3 
best_k = 3
best_model = models[Ks.index(best_k)]
labels_full = best_model.labels_
print(f"[OK] K fijado por criterio didáctico (exploración): {best_k}")

# 5) Reportes y guardados (exploración)
# -----------------------------
centroids_scaled = best_model.cluster_centers_
centroids_orig = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_orig, columns=features)
centroids_df.index = [f"Cluster {i+1}" for i in range(best_k)]
print("\n[Exploración] Centroides (unidades originales):\n", centroids_df.round(2))

df_out = df_clean.copy()
df_out["cluster_exploratorio_1aK"] = labels_full + 1  # SOLO exploración
summary = df_out.groupby("cluster_exploratorio_1aK")[features].mean().round(2)
summary["n_sesiones"] = df_out["cluster_exploratorio_1aK"].value_counts().sort_index().values

df_out[["row_idx_original","cluster_exploratorio_1aK"] + features].to_csv("garmin_kmeans_exploratorio.csv", index=False, encoding="utf-8")
summary.to_csv("garmin_kmeans_summary_exploratorio.csv", index=True, encoding="utf-8")
print("[OK] CSV exploratorio: garmin_kmeans_exploratorio.csv")
print("[OK] CSV resumen expl.: garmin_kmeans_summary_exploratorio.csv")

# 6) Interpretación de clusters (exploración, para rotular los grupos)
# -----------------------------
Z = centroids_df.copy().reset_index(drop=True)
for col in features:
    mu = Z[col].mean(); sd = Z[col].std() if Z[col].std() != 0 else 1.0
    Z[col] = (Z[col] - mu) / sd

long_score = Z["Distancia_km"] + Z["Tiempo_min"]
idx_long = int(np.argmax(long_score))
rest = [i for i in range(len(Z)) if i != idx_long]
intensity_score = (-Z.loc[rest, "Pace_min_km"].values) + Z.loc[rest, "FC_media_bpm"].values
idx_intense = rest[int(np.argmax(intensity_score))]
idx_short = [i for i in range(len(Z)) if i not in (idx_long, idx_intense)][0]

interpretaciones_expl = {
    f"Cluster {idx_short+1}":   "Sesiones cortas y suaves",
    f"Cluster {idx_long+1}":    "Entrenamientos largos y moderados",
    f"Cluster {idx_intense+1}": "Entrenamientos intensos / intervalos",
}
summary = summary.copy()
summary["Interpretacion"] = [interpretaciones_expl.get(f"Cluster {i}", f"Cluster {i}") for i in summary.index]

print("\n=== (Exploración) Etiquetas por clúster ===")
for i in range(1, best_k+1):
    print(f"Cluster {i}: {summary.loc[i, 'Interpretacion']}")


# 7) Gráficas 
# -----------------------------
cluster_colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
label_names_expl = {i: summary.loc[i, "Interpretacion"] for i in summary.index}

# 7.1 PCA 2D con centroides (exploración)
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
centroids_2d = pca.transform(centroids_scaled)

plt.figure()
for c in summary.index:
    pts = X_2d[(labels_full + 1) == c]
    plt.scatter(pts[:,0], pts[:,1], s=25, alpha=0.8,
                color=cluster_colors[c-1], label=f"C{c}: {label_names_expl[c]}")
plt.scatter(centroids_2d[:,0], centroids_2d[:,1], s=200, marker='*',
            edgecolor='black', linewidths=1.0, color='yellow', label='Centroides')
for i, (x, y) in enumerate(centroids_2d):
    plt.text(x, y, f"C{i+1}", fontsize=9, ha='center', va='bottom')
plt.title("Agrupamiento de sesiones (PCA 2D) - Exploración")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("pca_clusters_exploratorio.png", dpi=150)

# 7.2 Barras: tamaño por grupo (exploración)
plt.figure()
counts = summary["n_sesiones"].reindex(summary.index).values
labels_bar = [f"C{c}\n{label_names_expl[c]}" for c in summary.index]
plt.bar(labels_bar, counts, color=[cluster_colors[c-1] for c in summary.index])
for i, v in enumerate(counts):
    plt.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
plt.title("Número de sesiones por tipo (exploración)")
plt.ylabel("N° de sesiones")
plt.tight_layout(); plt.savefig("bar_tamano_exploratorio.png", dpi=150)

# 7.3 Boxplots (exploración)
for feat in ["Distancia_km","Tiempo_min","Pace_min_km"]:
    plt.figure()
    data_bp = [df_out.loc[df_out["cluster_exploratorio_1aK"]==c, feat].values for c in summary.index]
    plt.boxplot(data_bp, tick_labels=[f"C{c}" for c in summary.index], showfliers=False)
    plt.title(f"{feat} por cluster (exploración)")
    plt.ylabel(feat); plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(f"box_{feat}_exploratorio.png", dpi=150)

plt.show()


# =============================================================
# 8) Pipeline SUPERVISADO: split -> escalar(train) -> KMeans(train) -> etiquetas -> LR
# =============================================================
print("\n" + "="*70)
print("PIPELINE SUPERVISADO")
print("="*70)

# 8.1 Split (83/17 sobre 228 ≈ 189 train, 39 test)
X_train, X_test = train_test_split(X, test_size=39, random_state=42, shuffle=True)
print(f"[Split] Train={X_train.shape[0]} | Test={X_test.shape[0]} | Total={X.shape[0]}")

# 8.2 Escalado sin fuga
scaler_sup = StandardScaler()
Xtr = scaler_sup.fit_transform(X_train)
Xte = scaler_sup.transform(X_test)

# 8.3 K-Means en train (K=3)
kmeans_sup = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=42)
kmeans_sup.fit(Xtr)
ytr = kmeans_sup.predict(Xtr) + 1
yte = kmeans_sup.predict(Xte) + 1

# 8.4 Clasificador: Regresión Logística
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(Xtr, ytr)

yp = clf.predict(Xte)
yp_proba = clf.predict_proba(Xte)

# 8.5 Métricas
acc = accuracy_score(yte, yp)
prec, rec, f1, _ = precision_recall_fscore_support(yte, yp, labels=[1,2,3], average='macro')

print("\nRESULTADOS SUPERVISADOS (TEST)")
print(f"  Accuracy : {acc:.3f}")
print(f"  Precision: {prec:.3f}")
print(f"  Recall   : {rec:.3f}")
print(f"  F1-score : {f1:.3f}")

# Métricas por clase
print("\n[Métricas por clase]")
prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(yte, yp, labels=[1,2,3], zero_division=0)
for i, c in enumerate([1,2,3]):
    print(f"  Clase {c}: P={prec_c[i]:.3f} | R={rec_c[i]:.3f} | F1={f1_c[i]:.3f} | N={sup_c[i]}")

# 8.6 Matriz de confusión
cm = confusion_matrix(yte, yp, labels=[1, 2, 3])
fig = px.imshow(
    cm, text_auto=True,
    labels=dict(x="Predicción", y="Etiqueta verdadera", color="Conteo"),
    x=[f"C{c}" for c in [1,2,3]], y=[f"C{c}" for c in [1,2,3]],
    title="Matriz de confusión - Regresión Logística"
)
fig.update_layout(xaxis_side="bottom", margin=dict(l=60, r=20, t=60, b=60))
fig.update_yaxes(autorange="reversed")
fig.show()
fig.write_html("matriz_confusion_lr.html")

# 8.7 ROC AUC
yte_bin = np.eye(3)[yte-1]
auc_macro = roc_auc_score(yte_bin, yp_proba, multi_class="ovr", average="macro")
print(f"\nROC / AUC (macro, OVR): {auc_macro:.3f}")
