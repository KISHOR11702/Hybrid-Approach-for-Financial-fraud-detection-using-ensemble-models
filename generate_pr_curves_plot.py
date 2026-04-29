import os
import sys

# Ensure this is executed from d:\Major_project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Import app.py functions for identical preprocessing
from app import preprocess_data, preprocess_sequences_lstm_data, load_model, load_lstm_model
import app

def main():
    print("Loading models via app.py...")
    load_model()
    load_lstm_model()
    
    print("Loading dataset (test_dataset_10k.csv)...")
    df = pd.read_csv('test_dataset_10k.csv')
    
    if 'is_fraud' in df.columns:
        y_true = df['is_fraud'].values
    elif 'isFraud' in df.columns:
        y_true = df['isFraud'].values
    else:
        raise ValueError("Target column not found in dataset")
        
    print("Pre-processing for XGBoost...")
    X_xgb, feat_cols, df_clean = preprocess_data(df)
    
    print("Predicting with XGBoost...")
    xgb_probs = app.xgboost_model.predict_proba(X_xgb)[:, 1]
    
    print("Pre-processing for LSTM...")
    X_lstm, row_indices, df_processed = preprocess_sequences_lstm_data(df)
    
    print("Scaling LSTM sequences...")
    sh = X_lstm.shape
    if app.lstm_scaler is not None:
        X_lstm_sc = app.lstm_scaler.transform(X_lstm.reshape(-1, sh[-1])).reshape(sh)
    else:
        X_lstm_sc = X_lstm
        
    print("Predicting with LSTM...")
    lstm_probs_user = app.lstm_model.predict(X_lstm_sc, verbose=0).flatten()
    
    print("Mapping LSTM predictions to transaction level...")
    lstm_probs = np.zeros(len(df))
    for i, indices in enumerate(row_indices):
        for idx in indices:
            lstm_probs[idx] = lstm_probs_user[i]
            
    print("Calculating PR curve coordinates...")
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_true, xgb_probs)
    ap_xgb = average_precision_score(y_true, xgb_probs)
    
    precision_lstm, recall_lstm, _ = precision_recall_curve(y_true, lstm_probs)
    ap_lstm = average_precision_score(y_true, lstm_probs)
    
    print(f"XGBoost AP: {ap_xgb:.4f}")
    print(f"LSTM AP: {ap_lstm:.4f}")
    
    print("Plotting PR curves...")
    plt.figure(figsize=(10, 8))
    
    # Plot curves
    plt.plot(recall_xgb, precision_xgb, color='blue', lw=2, label=f'XGBoost (AP = {ap_xgb:.4f})')
    plt.plot(recall_lstm, precision_lstm, color='red', lw=2, label=f'LSTM (AP = {ap_lstm:.4f})')
    
    # Plot configuration
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves: XGBoost vs LSTM', fontsize=14)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/pr_curves.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ PR curves successfully generated and saved to {out_path}")

if __name__ == "__main__":
    main()
