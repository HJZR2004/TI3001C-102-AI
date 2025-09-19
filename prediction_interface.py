import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from model_training import train_logistic_regression_model, train_kmeans_model

# Global variables to store trained models
trained_models = {}

def load_trained_models():
    """Load the pre-trained models using the training functions"""
    global trained_models
    
    try:
        print("Loading trained models using training functions...")
        
        # Train logistic regression model
        print("Training logistic regression model...")
        logreg_components = train_logistic_regression_model()
        
        # Train k-means model  
        print("Training k-means model...")
        kmeans_components = train_kmeans_model()
        
        # Store all components
        trained_models = {
            'logistic_model': logreg_components['model'],
            'label_encoder': logreg_components['label_encoder'],
            'scaler': logreg_components['scaler'],
            'pca_logreg': logreg_components['pca'],
            'kmeans_model': kmeans_components['kmeans_model'],
            'kmeans_scaler': kmeans_components['scaler'],
            'pca_viz': kmeans_components['pca_viz'],
            'feature_names': logreg_components['feature_names']
        }
        
        print("Models loaded successfully using training functions!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False
# Global variables for GUI components
root = None
entries = {}
results_labels = {}

def create_interface():
    """Create the GUI interface"""
    global root, entries, results_labels
    
    root = tk.Tk()
    root.title("Activity Prediction System")
    root.geometry("600x500")
    
    # Title
    title_label = tk.Label(root, text="Activity Prediction System", 
                          font=("Arial", 16, "bold"))
    title_label.pack(pady=10)
    
    # Instructions
    instructions = tk.Label(root, 
                           text="Enter the activity data below to predict the type and cluster:",
                           font=("Arial", 10))
    instructions.pack(pady=5)
    
    # Create input frame
    input_frame = ttk.Frame(root)
    input_frame.pack(pady=20, padx=20, fill="x")
    
    # Input fields
    fields = [
        ("Distancia (km)", "distancia"),
        ("Calorías", "calorias"),
        ("Tiempo (min)", "tiempo"),
        ("Frecuencia Cardíaca Media", "frecuencia_cardiaca_media"),
        ("FC Máxima", "fc_maxima"),
        ("Tiempo en Movimiento (min)", "tiempo_en_movimiento")
    ]
    
    for i, (label_text, field_name) in enumerate(fields):
        label = ttk.Label(input_frame, text=label_text + ":")
        label.grid(row=i, column=0, sticky="w", padx=5, pady=5)
        
        entry = ttk.Entry(input_frame, width=20)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries[field_name] = entry
    
    # Predict button
    predict_button = ttk.Button(root, text="Predict", command=make_prediction)
    predict_button.pack(pady=20)
    
    # Results frame
    results_frame = ttk.LabelFrame(root, text="Prediction Results", padding=10)
    results_frame.pack(pady=10, padx=20, fill="both", expand=True)
    
    # Results labels
    results_labels['logistic_result'] = tk.Label(results_frame, text="", font=("Arial", 12, "bold"))
    results_labels['logistic_result'].pack(pady=5)
    
    results_labels['logistic_proba'] = tk.Label(results_frame, text="", font=("Arial", 10))
    results_labels['logistic_proba'].pack(pady=5)
    
    results_labels['kmeans_result'] = tk.Label(results_frame, text="", font=("Arial", 12, "bold"))
    results_labels['kmeans_result'].pack(pady=5)
    
    results_labels['cluster_info'] = tk.Label(results_frame, text="", font=("Arial", 10))
    results_labels['cluster_info'].pack(pady=5)
    
    # Clear button
    clear_button = ttk.Button(root, text="Clear", command=clear_inputs)
    clear_button.pack(pady=10)

def make_prediction():
    """Make predictions based on user input"""
    global trained_models, entries, results_labels
    
    try:
        # Get input values
        input_data = []
        for field_name in ["distancia", "calorias", "tiempo", 
                          "frecuencia_cardiaca_media", "fc_maxima", "tiempo_en_movimiento"]:
            value = float(entries[field_name].get())
            input_data.append(value)
        
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input using logistic regression scaler
        input_scaled = trained_models['scaler'].transform(input_array)
        
        # Logistic Regression Prediction
        input_pca = trained_models['pca_logreg'].transform(input_scaled)
        logreg_pred = trained_models['logistic_model'].predict(input_pca)[0]
        logreg_proba = trained_models['logistic_model'].predict_proba(input_pca)[0]
        
        # Get class name
        predicted_activity = trained_models['label_encoder'].inverse_transform([logreg_pred])[0]
        
        # K-Means Prediction (use kmeans scaler)
        input_scaled_kmeans = trained_models['kmeans_scaler'].transform(input_array)
        kmeans_pred = trained_models['kmeans_model'].predict(input_scaled_kmeans)[0]
        
        # Get distance to centroid
        distances = trained_models['kmeans_model'].transform(input_scaled_kmeans)[0]
        distance_to_cluster = distances[kmeans_pred]
        
        # Update results
        results_labels['logistic_result'].config(
            text=f"Predicted Activity: {predicted_activity}",
            fg="blue"
        )
        
        # Show probabilities
        proba_text = "Probabilities: "
        for i, class_name in enumerate(trained_models['label_encoder'].classes_):
            proba_text += f"{class_name}: {logreg_proba[i]:.3f}  "
        results_labels['logistic_proba'].config(text=proba_text, fg="gray")
        
        results_labels['kmeans_result'].config(
            text=f"Predicted Cluster: {kmeans_pred}",
            fg="green"
        )
        
        # Cluster interpretation
        cluster_descriptions = {
            0: "Low intensity activities",
            1: "Moderate intensity activities", 
            2: "High intensity activities"
        }
        
        cluster_desc = cluster_descriptions.get(kmeans_pred, "Unknown cluster type")
        results_labels['cluster_info'].config(
            text=f"Cluster Type: {cluster_desc}\nDistance to centroid: {distance_to_cluster:.3f}",
            fg="darkgreen"
        )
        
    except ValueError as e:
        messagebox.showerror("Input Error", "Please enter valid numerical values for all fields.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def clear_inputs():
    """Clear all input fields and results"""
    global entries, results_labels
    
    for entry in entries.values():
        entry.delete(0, tk.END)
    
    for label in results_labels.values():
        label.config(text="")

def run_gui():
    """Start the GUI application"""
    global root
    
    if load_trained_models():
        create_interface()
        root.mainloop()
    else:
        print("Failed to load models. Cannot start GUI.")

# Console version using training functions
def predict_from_console():
    """Alternative console-based prediction function"""
    print("=== Activity Prediction Console ===")
    print("Enter the following values:")
    
    try:
        # Load trained models using training functions
        if not load_trained_models():
            print("Failed to load models")
            return
        
        # Get user input
        distancia = float(input("Distancia (km): "))
        calorias = float(input("Calorías: "))
        tiempo = float(input("Tiempo (min): "))
        fc_media = float(input("Frecuencia Cardíaca Media: "))
        fc_maxima = float(input("FC Máxima: "))
        tiempo_movimiento = float(input("Tiempo en Movimiento (min): "))
        
        # Make predictions
        input_data = np.array([[distancia, calorias, tiempo, fc_media, fc_maxima, tiempo_movimiento]])
        
        # Scale using logistic regression scaler
        input_scaled = trained_models['scaler'].transform(input_data)
        
        # Logistic Regression
        input_pca = trained_models['pca_logreg'].transform(input_scaled)
        logreg_pred = trained_models['logistic_model'].predict(input_pca)[0]
        logreg_proba = trained_models['logistic_model'].predict_proba(input_pca)[0]
        predicted_activity = trained_models['label_encoder'].inverse_transform([logreg_pred])[0]
        
        # K-Means (use kmeans scaler)
        input_scaled_kmeans = trained_models['kmeans_scaler'].transform(input_data)
        kmeans_pred = trained_models['kmeans_model'].predict(input_scaled_kmeans)[0]
        
        # Results
        print(f"\n=== PREDICTION RESULTS ===")
        print(f"Predicted Activity Type: {predicted_activity}")
        print(f"Predicted Cluster: {kmeans_pred}")
        
        print(f"\nProbabilities:")
        for i, class_name in enumerate(trained_models['label_encoder'].classes_):
            print(f"  {class_name}: {logreg_proba[i]:.3f}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run the GUI version
    run_gui()
    
    # Uncomment the line below to run the console version instead
    # predict_from_console()