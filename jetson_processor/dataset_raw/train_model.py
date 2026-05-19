#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle

# 1. Charger les données
print("Chargement du dataset...")
df = pd.read_csv('features_dataset_v3.csv')

# 2. Définir les features
feature_cols = [
    'packet_count', 'duration', 'packet_rate',
    'rssi_mean', 'rssi_std', 'rssi_min', 'rssi_max', 'rssi_range',
    'rssi_median', 'rssi_q25', 'rssi_q75', 'rssi_iqr',
    'rssi_diff_mean', 'rssi_diff_abs_mean', 'rssi_diff_std', 'rssi_diff_max_abs',
    'len_mean', 'len_std', 'len_min', 'len_max',
    'count_gap_mean', 'count_gap_std', 'count_gap_max',
    'time_gap_mean', 'time_gap_std', 'time_gap_max'
]

# Standardisation simple (indispensable pour le k-NN)
X = df[feature_cols].values
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = df['label_id'].values
labels = df['label'].values

# 3. Séparation simple (80% train, 20% test)
indices = np.arange(len(df))
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.8 * len(df))

train_idx, test_idx = indices[:split], indices[split:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 4. Prédiction k-NN (k=3)
def predict_knn(X_train, y_train, X_test, k=3):
    preds = []
    for test_row in X_test:
        distances = np.sqrt(np.sum((X_train - test_row)**2, axis=1))
        nearest = y_train[np.argsort(distances)[:k]]
        preds.append(np.bincount(nearest).argmax())
    return np.array(preds)

print("Calcul des prédictions (k-NN)...")
y_pred = predict_knn(X_train, y_train, X_test, k=3)

# 5. Évaluation simple
print("\n--- Résultats (k-NN k=3) ---")
correct = np.sum(y_pred == y_test)
print("Accuracy: %.2f%%" % (correct / len(y_test) * 100))

# 6. Sauvegarde du modèle (les données d'entraînement)
model_filename = 'csi_knn_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump({'X_train': X_train, 'y_train': y_train, 'feature_cols': feature_cols}, f)
print("\nModèle (données d'entraînement) sauvegardé sous : %s" % model_filename)
