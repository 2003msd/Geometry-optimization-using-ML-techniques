import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

tf.random.set_seed(42)
np.random.seed(42)

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath).dropna()
    X = df[['n', 'Diameter (m)', 'Thickness (m)']].values
    y = df[['Alpha1', 'Bcon1', 'Ccon1']].values

    
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

 
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2857, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler

def build_ann(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mse = mean_squared_error(y_true, y_pred)
    mean_val = np.mean(np.abs(y_true))
    acc = 100 * (1 - mae / mean_val)
    return mae, mape, mse, acc

def train_and_display(model, X_train, y_train, X_val, y_val, y_scaler, epochs=200, display_every=10):
    for epoch in range(1, epochs + 1):
        model.fit(X_train, y_train, batch_size=16, epochs=1, verbose=0)

        train_pred = y_scaler.inverse_transform(model.predict(X_train, verbose=0))
        val_pred = y_scaler.inverse_transform(model.predict(X_val, verbose=0))
        y_train_true = y_scaler.inverse_transform(y_train)
        y_val_true = y_scaler.inverse_transform(y_val)
        
        print(f"\nEpoch {epoch}")
        for i, label in enumerate(['Alpha1', 'Bcon1', 'Ccon1']):
            t_mae, t_mape, _, t_acc = compute_metrics(y_train_true[:, i], train_pred[:, i])
            v_mae, v_mape, _, v_acc = compute_metrics(y_val_true[:, i], val_pred[:, i])
            print(f"{label:<7} | Train → MAE: {t_mae:.4f}, MAPE: {t_mape:.2f}%, Accuracy: {t_acc:.2f}%")
            print(f"{'':7} | Val   → MAE: {v_mae:.4f}, MAPE: {v_mape:.2f}%, Accuracy: {v_acc:.2f}%")
        print("-" * 70)

def evaluate_and_display(model, X_test, y_test, y_scaler):
    y_pred = y_scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_true = y_scaler.inverse_transform(y_test)

    print("\n✅ Final Test Evaluation:\n")
    for i, label in enumerate(['Alpha1', 'Bcon1', 'Ccon1']):
        mae, mape, mse, acc = compute_metrics(y_true[:, i], y_pred[:, i])
        print(f"=== {label} ===")
        print(f"MAE     : {mae:.4f}")
        print(f"MAPE    : {mape:.2f}%")
        print(f"MSE     : {mse:.4f}")
        print(f"Accuracy: {acc:.2f}%\n")

if __name__ == "__main__":
    filepath = "../Dataset/arch_optimization_final_results.csv"
    X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler = load_and_prepare_data(filepath)

    model = build_ann(input_dim=3, output_dim=3)
    train_and_display(model, X_train, y_train, X_val, y_val, y_scaler, epochs=200, display_every=10)

    evaluate_and_display(model, X_test, y_test, y_scaler)
