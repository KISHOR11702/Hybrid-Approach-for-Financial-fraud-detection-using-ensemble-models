"""
06_lstm_final.py
Loads the best LSTM sequences checkpoint and evaluates / exports final model.
Also runs grid search over threshold values and saves the optimal one.
"""

import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, confusion_matrix,
                              f1_score, precision_score, recall_score)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras

MODELS_DIR = '../models'
LSTM_TRAINING_DIR = '../lstm_training_data'

BEST_MODEL  = f'{MODELS_DIR}/lstm_transaction_sequences_best.h5'
FINAL_MODEL = f'{MODELS_DIR}/lstm_transaction_sequences_final.h5'
SCALER_PATH = f'{MODELS_DIR}/lstm_sequences_scaler.pkl'
THRESH_PATH = f'{MODELS_DIR}/lstm_sequences_threshold.txt'


def load_test_data():
    X_test = np.load(f'{LSTM_TRAINING_DIR}/X_test_lstm.npy')
    y_test = np.load(f'{LSTM_TRAINING_DIR}/y_test_lstm.npy')
    return X_test, y_test


if __name__ == '__main__':
    print("Loading best LSTM model ...")
    model = keras.models.load_model(BEST_MODEL)
    model.summary()

    # Load or generate test data
    try:
        X_test, y_test = load_test_data()
        print(f"Loaded test data: {X_test.shape}")

        # Apply scaler if available
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            sh = X_test.shape
            X_test = scaler.transform(X_test.reshape(-1, sh[-1])).reshape(sh)
            print("Applied scaler to test data.")

        y_prob = model.predict(X_test, verbose=0).flatten()

    except FileNotFoundError:
        print("⚠️  Test data not found in lstm_training_data/. Run transaction_sequences_lstm.py first.")
        y_prob = None
        y_test = None

    if y_prob is not None:
        # ── Threshold grid search ─────────────────────────────────────────
        best_f1, best_thr = 0, 0.5
        for thr in np.arange(0.3, 0.9, 0.01):
            preds = (y_prob >= thr).astype(int)
            f1    = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        print(f"\nBest threshold by F1: {best_thr:.2f}  (F1={best_f1:.4f})")

        y_pred  = (y_prob >= best_thr).astype(int)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc  = average_precision_score(y_test, y_prob)

        print(f"Precision : {precision_score(y_test,y_pred,zero_division=0)*100:.2f}%")
        print(f"Recall    : {recall_score(y_test,y_pred,zero_division=0)*100:.2f}%")
        print(f"F1-Score  : {best_f1*100:.2f}%")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"PR-AUC    : {pr_auc:.4f}")
        print(f"Confusion :\n{confusion_matrix(y_test, y_pred)}")

        with open(THRESH_PATH, 'w') as f:
            f.write(f'{best_thr:.4f}')
        print(f"✅ Saved threshold {best_thr:.4f} → {THRESH_PATH}")

    # ── Re-save as final ──────────────────────────────────────────────────
    model.save(FINAL_MODEL)
    print(f"✅ Saved final model → {FINAL_MODEL}")
