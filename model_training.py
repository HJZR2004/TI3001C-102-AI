"""
Model training module that provides training functions for both models
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def train_logistic_regression_model():
    """
    Function to train the logistic regression model and return all necessary components
    for making predictions on new data.
    
    Returns:
        dict: Dictionary containing trained model and preprocessing components
    """
    # Load data
    df = pd.read_csv(r"Data\Processed_Activities.csv")
    
    # Variables predictoras y objetivo
    X = df[["distancia", "calorias", "tiempo",
            "frecuencia_cardiaca_media", "fc_maxima", "tiempo_en_movimiento"]]
    y = df["tipo_de_actividad"]
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for dimensionality reduction (retain 95% variance)
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Train/test split on PCA features
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, y_encoded, test_size=0.17, random_state=42, stratify=y_encoded
    )
    
    # Logistic Regression Model with PCA features and class_weight balanced
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_pca, y_train)
    
    return {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'pca': pca,
        'feature_names': ["distancia", "calorias", "tiempo",
                         "frecuencia_cardiaca_media", "fc_maxima", "tiempo_en_movimiento"]
    }

def train_kmeans_model():
    """
    Function to train the K-means model and return all necessary components
    for making predictions on new data.
    
    Returns:
        dict: Dictionary containing trained model and preprocessing components
    """
    # Load data
    df = pd.read_csv(r"Data\Processed_Activities.csv")
    
    # Variables numéricas (sin incluir la etiqueta)
    X = df.drop(columns=["tipo_de_actividad"])
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for visualization (2 components)
    pca_viz = PCA(n_components=2, random_state=42)
    X_pca = pca_viz.fit_transform(X_scaled)
    
    # Entrenar con k clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    return {
        'kmeans_model': kmeans,
        'scaler': scaler,
        'pca_viz': pca_viz,
        'feature_names': X.columns.tolist()
    }