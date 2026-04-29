"""
08_meta_model_hybrid.py
Trains a Logistic Regression meta-learner on top of XGBoost + LSTM output
probabilities (stacking ensemble). Improves precision over the soft-vote hybrid.
"""

import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              confusion_matrix)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras

MODELS_DIR        = '../models'
LSTM_TRAINING_DIR = '../lstm_training_data'
PREPROCESSED_DIR  = '../preprocessed_data_no_leakage'


def load_models():
    with open(f'{MODELS_DIR}/xgboost_model_no_leakage.pkl', 'rb') as f:
        xgb = pickle.load(f)
    lstm = keras.models.load_model(f'{MODELS_DIR}/lstm_transaction_sequences_final.h5')
    with open(f'{MODELS_DIR}/lstm_sequences_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return xgb, lstm, scaler


if __name__ == '__main__':
    xgb_model, lstm_model, lstm_scaler = load_models()

    # Load base probabilities from training set
    X_train_xgb  = np.load(f'{PREPROCESSED_DIR}/X_train_scaled.npy')
    X_train_lstm = np.load(f'{LSTM_TRAINING_DIR}/X_train_lstm.npy')
    y_train      = np.load(f'{PREPROCESSED_DIR}/y_train.npy')

    X_test_xgb   = np.load(f'{PREPROCESSED_DIR}/X_test_scaled.npy')
    X_test_lstm  = np.load(f'{LSTM_TRAINING_DIR}/X_test_lstm.npy')
    y_test       = np.load(f'{PREPROCESSED_DIR}/y_test.npy')

    print(f"Train: {len(y_train)}  Test: {len(y_test)}")

    # Generate base probabilities
    def get_probs(xgb, lstm, scaler, X_xgb, X_lstm):
        xgb_p  = xgb.predict_proba(X_xgb)[:, 1]
        sh     = X_lstm.shape
        X_sc   = scaler.transform(X_lstm.reshape(-1, sh[-1])).reshape(sh)
        lstm_p = lstm.predict(X_sc, verbose=0).flatten()
        n      = min(len(xgb_p), len(lstm_p))
        return np.column_stack([xgb_p[:n], lstm_p[:n]])

    print("Generating train-set probabilities ...")
    Z_train = get_probs(xgb_model, lstm_model, lstm_scaler,
                        X_train_xgb, X_train_lstm)
    n_tr    = min(len(Z_train), len(y_train))
    y_tr    = y_train[:n_tr]

    print("Generating test-set probabilities ...")
    Z_test  = get_probs(xgb_model, lstm_model, lstm_scaler,
                        X_test_xgb, X_test_lstm)
    n_te    = min(len(Z_test), len(y_test))
    y_te    = y_test[:n_te]

    # ── Meta-learner ──────────────────────────────────────────────────────
    meta = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
    cv_scores = cross_val_score(meta, Z_train[:n_tr], y_tr,
                                cv=5, scoring='roc_auc')
    print(f"\nMeta CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    meta.fit(Z_train[:n_tr], y_tr)
    meta_probs = meta.predict_proba(Z_test[:n_te])[:, 1]
    meta_preds = meta.predict(Z_test[:n_te])

    roc_auc  = roc_auc_score(y_te, meta_probs)
    pr_auc   = average_precision_score(y_te, meta_probs)
    prec     = precision_score(y_te, meta_preds, zero_division=0)
    rec      = recall_score(y_te, meta_preds,    zero_division=0)
    f1       = f1_score(y_te, meta_preds,        zero_division=0)
    tn,fp,fn,tp = confusion_matrix(y_te, meta_preds).ravel()

    print(f"\n=== Meta-Learner Results ===")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={prec*100:.2f}%  Recall={rec*100:.2f}%  F1={f1*100:.2f}%")
    print(f"  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    meta_path = f'{MODELS_DIR}/meta_model_hybrid.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"\n✅ Saved meta-learner → {meta_path}")
