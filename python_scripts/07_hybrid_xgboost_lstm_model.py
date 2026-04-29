"""
07_hybrid_xgboost_lstm_model.py
Combines XGBoost and LSTM predictions via weighted soft-vote ensemble.
  XGBoost weight = 0.40
  LSTM     weight = 0.60
  Combined threshold = 0.55
Evaluates the hybrid on the held-out test set and reports metrics.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix, f1_score,
                              precision_score, recall_score,
                              precision_recall_curve)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras

MODELS_DIR        = '../models'
LSTM_TRAINING_DIR = '../lstm_training_data'
PREPROCESSED_DIR  = '../preprocessed_data_no_leakage'

XGB_WEIGHT    = 0.40
LSTM_WEIGHT   = 0.60
HYBRID_THRESH = 0.55


def load_xgboost():
    with open(f'{MODELS_DIR}/xgboost_model_no_leakage.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{MODELS_DIR}/xgboost_model_no_leakage_threshold.pkl', 'rb') as f:
        thresh = pickle.load(f)['threshold']
    print(f"✅ XGBoost loaded  (threshold={thresh})")
    return model, thresh


def load_lstm():
    model = keras.models.load_model(
        f'{MODELS_DIR}/lstm_transaction_sequences_final.h5')
    with open(f'{MODELS_DIR}/lstm_sequences_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{MODELS_DIR}/lstm_sequences_threshold.txt') as f:
        thresh = float(f.read().strip())
    print(f"✅ LSTM loaded  (threshold={thresh})")
    return model, scaler, thresh


def evaluate_hybrid(y_true, xgb_probs, lstm_probs, hybrid_thresh=HYBRID_THRESH,
                    xgb_thresh=0.60, lstm_thresh=0.63):
    combined    = XGB_WEIGHT * xgb_probs + LSTM_WEIGHT * lstm_probs
    hybrid_preds = (combined >= hybrid_thresh).astype(int)
    xgb_preds    = (xgb_probs  >= xgb_thresh).astype(int)
    lstm_preds   = (lstm_probs >= lstm_thresh).astype(int)

    for name, preds, probs in [
        ('XGBoost', xgb_preds, xgb_probs),
        ('LSTM',    lstm_preds, lstm_probs),
        ('HYBRID',  hybrid_preds, combined),
    ]:
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        prec   = precision_score(y_true, preds, zero_division=0)
        rec    = recall_score(y_true, preds,    zero_division=0)
        f1     = f1_score(y_true, preds,        zero_division=0)
        auc    = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
        print(f"\n── {name} ──")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Precision={prec*100:.2f}%  Recall={rec*100:.2f}%"
              f"  F1={f1*100:.2f}%  ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")


if __name__ == '__main__':
    # Load models
    xgb_model, xgb_thresh = load_xgboost()
    lstm_model, lstm_scaler, lstm_thresh = load_lstm()

    # Load test data
    X_test_xgb = np.load(f'{PREPROCESSED_DIR}/X_test_scaled.npy')
    X_test_lstm = np.load(f'{LSTM_TRAINING_DIR}/X_test_lstm.npy')
    y_test      = np.load(f'{PREPROCESSED_DIR}/y_test.npy')
    print(f"\nTest samples: {len(y_test)}  Fraud: {y_test.sum()}")

    # XGBoost probabilities
    xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]

    # LSTM probabilities (per-user, mapped back to rows if needed)
    sh_lstm   = X_test_lstm.shape
    X_lstm_sc = lstm_scaler.transform(
        X_test_lstm.reshape(-1, sh_lstm[-1])).reshape(sh_lstm)
    lstm_probs = lstm_model.predict(X_lstm_sc, verbose=0).flatten()

    # Align lengths (take min in edge cases)
    n = min(len(xgb_probs), len(lstm_probs), len(y_test))
    xgb_probs, lstm_probs, y_test = xgb_probs[:n], lstm_probs[:n], y_test[:n]

    print("\n" + "=" * 60)
    print("HYBRID ENSEMBLE EVALUATION")
    print("=" * 60)
    print(f"Weights:   XGBoost={XGB_WEIGHT}  LSTM={LSTM_WEIGHT}")
    print(f"Threshold: {HYBRID_THRESH}")
    evaluate_hybrid(y_test, xgb_probs, lstm_probs, HYBRID_THRESH,
                    xgb_thresh, lstm_thresh)
