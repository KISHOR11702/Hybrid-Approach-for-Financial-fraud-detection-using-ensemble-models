"""
05_lstm_model_no_leakage.py
Trains an LSTM model on PaySim data with no-leakage features.
Uses step-based sequences per account (nameOrig) with 15-step windows.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_PATH  = '../engineered_data.csv'
MODELS_DIR = '../models'
os.makedirs(MODELS_DIR, exist_ok=True)

SEQ_LEN    = 15
BATCH_SIZE = 256
EPOCHS     = 30
SEED       = 42
np.random.seed(SEED)

# PaySim features available for LSTM sequences
SEQ_FEATURES = [
    'amount', 'logAmount', 'isHighRiskType', 'isTransfer', 'isCashOut',
    'isC2C', 'isC2M', 'hourOfDay', 'isNightTime', 'dayOfWeek',
    'isWeekend', 'isHighAmount', 'typeEncoded', 'origIsCustomer', 'destIsCustomer'
]
TARGET = 'isFraud'


def build_sequences_paysim(df, seq_len=SEQ_LEN):
    """Build per-account LSTM sequences from PaySim data sorted by step."""
    df = df.sort_values(['nameOrig', 'step']).reset_index(drop=True)
    X_seqs, y_labels = [], []
    n_feat = len(SEQ_FEATURES)

    for uid, grp in df.groupby('nameOrig', sort=False):
        feats = grp[SEQ_FEATURES].values.astype(float)
        label = int(grp[TARGET].max())
        n     = len(feats)
        if n >= seq_len:
            seq = feats[-seq_len:]
        else:
            seq = np.vstack([np.zeros((seq_len - n, n_feat)), feats])
        X_seqs.append(seq)
        y_labels.append(label)

    return np.array(X_seqs), np.array(y_labels)


def build_model(seq_len, n_feats):
    inp = keras.Input(shape=(seq_len, n_feats))
    x   = layers.LSTM(64, return_sequences=True,  dropout=0.2)(inp)
    x   = layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
    x   = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1,  activation='sigmoid')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


if __name__ == '__main__':
    print("Loading data ...")
    df = pd.read_csv(DATA_PATH)
    for col in SEQ_FEATURES:
        if col not in df.columns:
            df[col] = 0

    print("Building per-account sequences ...")
    X, y = build_sequences_paysim(df)
    print(f"Sequences: {X.shape}  Fraud: {y.sum()}/{len(y)}")

    sh     = X.shape
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X.reshape(-1, sh[-1])).reshape(sh)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=SEED, stratify=y)

    fraud_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model   = build_model(SEQ_LEN, len(SEQ_FEATURES))
    model.summary()

    model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight={0: 1.0, 1: fraud_w},
        callbacks=[
            EarlyStopping(monitor='val_AUC', patience=8,
                          restore_best_weights=True, mode='max'),
            ModelCheckpoint(f'{MODELS_DIR}/lstm_no_leakage_model.h5',
                            monitor='val_AUC', save_best_only=True, mode='max'),
        ],
        verbose=1
    )

    y_prob  = model.predict(X_te, verbose=0).flatten()
    roc_auc = roc_auc_score(y_te, y_prob)
    pr_auc  = average_precision_score(y_te, y_prob)
    print(f"\nROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")

    with open(f'{MODELS_DIR}/lstm_no_leakage_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Done. Model saved to models/lstm_no_leakage_model.h5")
