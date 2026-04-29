"""
transaction_sequences_lstm.py
Trains the leakage-free LSTM sequences model (seq_len=15, 19 features).
Uses per-user behavioural sequences from the fraud dataset.
Saves: lstm_transaction_sequences_final.h5
       lstm_transaction_sequences_best.h5
       lstm_sequences_scaler.pkl
       lstm_sequences_threshold.txt
       lstm_sequences_metadata.json
       lstm_sequences_features.txt
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                              average_precision_score, f1_score,
                              precision_score, recall_score, confusion_matrix)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATA_PATH  = '../fraud_detection_dataset_lstm_v2.csv'
MODELS_DIR = '../models'
PLOTS_DIR  = '../plots'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

SEQ_LEN    = 15
BATCH_SIZE = 512
EPOCHS     = 50
SEED       = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── 19 sequence features ──────────────────────────────────────────────────────
SEQ_FEATURES = [
    'amount', 'merchant_category_encoded', 'time_gap_from_previous_hrs',
    'amount_deviation_from_avg_pct', 'transactions_last_1hr', 'transactions_last_24hr',
    'account_age_days', 'avg_transaction_amount', 'transaction_frequency',
    'device_id_encoded', 'location_encoded', 'most_used_device_encoded',
    'most_used_location_encoded', 'location_changed', 'device_changed',
    'hour', 'day_of_week', 'day_of_month', 'month'
]

MERCHANT_CATEGORIES = [
    'Grocery','Restaurant','Gas Station','Online Shopping','Electronics',
    'Pharmacy','Entertainment','Travel','Utilities','Insurance',
    'Healthcare','Education','Clothing','Home Improvement','ATM Withdrawal'
]
DEVICE_TYPES = [
    'Mobile_iOS','Mobile_Android','Desktop_Windows',
    'Desktop_Mac','Tablet_iOS','Tablet_Android','Web_Browser'
]
LOCATIONS = [
    'Mumbai','Delhi','Bangalore','Hyderabad','Chennai','Kolkata',
    'Pune','Ahmedabad','Jaipur','Lucknow','Surat','Chandigarh',
    'Bhopal','Nagpur','Vadodara','Coimbatore','Indore','Patna'
]


def encode_col(series, vocab):
    m = {v: i for i, v in enumerate(vocab)}
    return series.map(lambda x: m.get(str(x), 0))


def most_common_past(group_series):
    result, past = [], []
    for val in group_series:
        result.append(max(set(past), key=past.count) if past else val)
        past.append(val)
    return pd.Series(result, index=group_series.index)


def load_and_preprocess(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    df['hour']         = df['timestamp'].dt.hour.fillna(12).astype(int)
    df['day_of_week']  = df['timestamp'].dt.dayofweek.fillna(1).astype(int)
    df['day_of_month'] = df['timestamp'].dt.day.fillna(15).astype(int)
    df['month']        = df['timestamp'].dt.month.fillna(1).astype(int)

    df['merchant_category_encoded'] = encode_col(df['merchant_category'], MERCHANT_CATEGORIES)
    df['device_id_encoded']         = encode_col(df['device_id'],         DEVICE_TYPES)
    df['location_encoded']          = encode_col(df['location'],          LOCATIONS)

    df['most_used_device']   = df.groupby('user_id', group_keys=False)['device_id'].apply(most_common_past)
    df['most_used_location'] = df.groupby('user_id', group_keys=False)['location'].apply(most_common_past)

    df['most_used_device_encoded']   = encode_col(df['most_used_device'],   DEVICE_TYPES)
    df['most_used_location_encoded'] = encode_col(df['most_used_location'], LOCATIONS)

    df['location_changed'] = (df['location'] != df['most_used_location']).astype(int)
    df['device_changed']   = (df['device_id'] != df['most_used_device']).astype(int)

    for col in SEQ_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df[SEQ_FEATURES] = df[SEQ_FEATURES].fillna(0)

    return df


def build_sequences(df, seq_len=SEQ_LEN):
    """Build (N_users, SEQ_LEN, 19) sequences and per-user labels."""
    X_seqs, y_labels, user_ids = [], [], []
    n_feat = len(SEQ_FEATURES)

    for uid, grp in df.groupby('user_id', sort=False):
        feats  = grp[SEQ_FEATURES].values.astype(float)
        label  = int(grp['isFraud'].max())   # 1 if any transaction is fraud
        n      = len(feats)
        if n >= seq_len:
            seq = feats[-seq_len:]
        else:
            seq = np.vstack([np.zeros((seq_len - n, n_feat)), feats])
        X_seqs.append(seq)
        y_labels.append(label)
        user_ids.append(uid)

    return np.array(X_seqs), np.array(y_labels), user_ids


def build_model(seq_len, n_feats):
    inp = keras.Input(shape=(seq_len, n_feats))
    x   = layers.LSTM(128, return_sequences=True, dropout=0.3,
                      recurrent_dropout=0.2)(inp)
    x   = layers.LSTM(64,  return_sequences=False, dropout=0.3,
                      recurrent_dropout=0.2)(x)
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1,  activation='sigmoid')(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    return model


if __name__ == '__main__':
    df = load_and_preprocess(DATA_PATH)

    X, y, user_ids = build_sequences(df, SEQ_LEN)
    print(f"\nSequences: {X.shape}  Labels: {y.shape}")
    print(f"Fraud users: {y.sum()} / {len(y)} ({y.mean()*100:.2f}%)")

    # ── Scale ────────────────────────────────────────────────────────────
    sh     = X.shape
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X.reshape(-1, sh[-1]))
    X_sc   = X_flat.reshape(sh)

    with open(f'{MODELS_DIR}/lstm_sequences_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Saved lstm_sequences_scaler.pkl")

    # ── Train/test split ─────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # ── Class weight ─────────────────────────────────────────────────────
    fraud_w = (y_tr == 0).sum() / (y_tr == 1).sum()
    cw      = {0: 1.0, 1: fraud_w}
    print(f"Class weight for fraud: {fraud_w:.1f}")

    # ── Build & train ────────────────────────────────────────────────────
    model = build_model(SEQ_LEN, len(SEQ_FEATURES))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_AUC', patience=10, restore_best_weights=True,
                      mode='max', verbose=1),
        ModelCheckpoint(f'{MODELS_DIR}/lstm_transaction_sequences_best.h5',
                        monitor='val_AUC', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_AUC', factor=0.5, patience=5,
                          min_lr=1e-6, mode='max', verbose=1),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )

    model.save(f'{MODELS_DIR}/lstm_transaction_sequences_final.h5')
    print("✅ Saved lstm_transaction_sequences_final.h5")

    # ── Evaluate ─────────────────────────────────────────────────────────
    y_prob = model.predict(X_te, verbose=0).flatten()
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_te, y_prob)
    f1_arr  = np.where((prec_arr + rec_arr) == 0, 0,
                       2 * prec_arr * rec_arr / (prec_arr + rec_arr))
    best_i  = np.argmax(f1_arr)
    opt_thr = float(thr_arr[best_i]) if best_i < len(thr_arr) else 0.50

    y_pred   = (y_prob >= opt_thr).astype(int)
    roc_auc  = roc_auc_score(y_te, y_prob)
    pr_auc   = average_precision_score(y_te, y_prob)
    prec_val = precision_score(y_te, y_pred, zero_division=0)
    rec_val  = recall_score(y_te, y_pred,    zero_division=0)
    f1_val   = f1_score(y_te, y_pred,        zero_division=0)

    print(f"\n=== LSTM Evaluation (threshold={opt_thr:.4f}) ===")
    print(f"  Precision : {prec_val*100:.2f}%")
    print(f"  Recall    : {rec_val*100:.2f}%")
    print(f"  F1-Score  : {f1_val*100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  Confusion :\n{confusion_matrix(y_te, y_pred)}")

    # ── Save threshold & metadata ─────────────────────────────────────────
    with open(f'{MODELS_DIR}/lstm_sequences_threshold.txt', 'w') as f:
        f.write(f'{opt_thr:.4f}')

    metadata = {
        'seq_len': SEQ_LEN,
        'n_features': len(SEQ_FEATURES),
        'features': SEQ_FEATURES,
        'threshold': opt_thr,
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
    }
    with open(f'{MODELS_DIR}/lstm_sequences_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(f'{MODELS_DIR}/lstm_sequences_features.txt', 'w') as f:
        f.write('\n'.join(SEQ_FEATURES))

    print(f"\n✅ Saved threshold: {opt_thr:.4f}")
    print("✅ Saved lstm_sequences_metadata.json")
    print("✅ Training complete.")
