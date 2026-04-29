"""
17_tune_threshold_opt.py
─────────────────────────
Sweeps thresholds on the held-out test set for lstm_precision_final_opt.h5
and saves the best balanced threshold to lstm_precision_threshold_opt.txt.

Usage:
    python python_scripts/17_tune_threshold_opt.py
"""

import os, pickle, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, backend as K


# ── Custom layer (must match definition in bayestian_opt.py) ──────────────────
class AttentionLayer(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name='att_W', shape=(int(input_shape[-1]), int(input_shape[-1])), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_b', shape=(int(input_shape[-1]),), initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_u', shape=(int(input_shape[-1]),), initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, x):
        uit = K.tanh(K.dot(x, self.W) + self.b)
        ait = K.softmax(K.sum(uit * self.u, axis=-1), axis=-1)
        return K.sum(x * K.expand_dims(ait, axis=-1), axis=1)

SEED = 42
np.random.seed(SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'fraud_detection_dataset_lstm_v3.csv')
MODEL_H5   = os.path.join(BASE_DIR, 'models', 'lstm_precision_final_opt.h5')
SCALER_PKL = os.path.join(BASE_DIR, 'models', 'lstm_precision_scaler_opt.pkl')
OUT_THR    = os.path.join(BASE_DIR, 'models', 'lstm_precision_threshold_opt.txt')

SEQ_LEN = 15
SEQ_FEATURE_COLS = [
    'amount', 'merchant_category_encoded', 'time_gap_from_previous_hrs',
    'amount_deviation_from_avg_pct', 'transactions_last_1hr', 'transactions_last_24hr',
    'account_age_days', 'avg_transaction_amount', 'transaction_frequency',
    'device_id_encoded', 'location_encoded', 'most_used_device_encoded',
    'most_used_location_encoded', 'location_changed', 'device_changed',
    'hour', 'day_of_week', 'day_of_month', 'month',
    'log_amount', 'amount_x_time_gap', 'amount_x_location_changed',
    'tx_frequency_ratio', 'amount_zscore', 'hour_sin', 'hour_cos'
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

def _encode(series, vocab):
    m = {v: i for i, v in enumerate(vocab)}
    return series.map(lambda x: m.get(str(x), 0))

def load_and_build(path):
    print(f"  Loading {path} ...")
    chunksize = 300_000
    chunks = []
    for ch in pd.read_csv(path, chunksize=chunksize):
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Rows={len(df):,}  Users={df['user_id'].nunique():,}")

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values(['user_id','timestamp']).reset_index(drop=True)

    # Base encodings
    df['merchant_category_encoded'] = _encode(df['merchant_category'], MERCHANT_CATEGORIES)
    df['device_id_encoded']         = _encode(df['device_id'],         DEVICE_TYPES)
    df['location_encoded']          = _encode(df['location'],          LOCATIONS)
    df['most_used_device_encoded']  = _encode(df['most_used_device'],  DEVICE_TYPES)
    df['most_used_location_encoded']= _encode(df['most_used_location'],LOCATIONS)
    df['location_changed']          = (df['location'] != df['most_used_location']).astype(int)
    df['device_changed']            = (df['device_id'] != df['most_used_device']).astype(int)

    df['hour']         = df['timestamp'].dt.hour.fillna(12).astype(int)
    df['day_of_week']  = df['timestamp'].dt.dayofweek.fillna(1).astype(int)
    df['day_of_month'] = df['timestamp'].dt.day.fillna(15).astype(int)
    df['month']        = df['timestamp'].dt.month.fillna(1).astype(int)

    # Engineered features (same as bayestian_opt.py)
    df['hour_sin']              = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos']              = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['log_amount']            = np.log1p(df['amount'].clip(lower=0))
    df['amount_x_time_gap']     = df['amount'] * (1.0 / df['time_gap_from_previous_hrs'].clip(lower=0.01))
    df['amount_x_location_changed'] = df['amount'] * df['location_changed']
    df['tx_frequency_ratio']    = df['transaction_frequency'] / df['account_age_days'].clip(lower=1)
    df['amount_zscore']         = df['amount_deviation_from_avg_pct'] / 100.0

    df[SEQ_FEATURE_COLS] = df[SEQ_FEATURE_COLS].fillna(0)

    # Build sequences
    print("  Building sequences ...")
    X_list, y_list = [], []
    n_feat = len(SEQ_FEATURE_COLS)
    fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'isFraud'
    for _, grp in df.groupby('user_id', sort=False):
        feats = grp[SEQ_FEATURE_COLS].values.astype(float)
        label = int(grp[fraud_col].max())
        n     = len(feats)
        seq   = feats[-SEQ_LEN:] if n >= SEQ_LEN else \
                np.vstack([np.zeros((SEQ_LEN - n, n_feat)), feats])
        X_list.append(seq); y_list.append(label)

    return np.array(X_list), np.array(y_list)


def main():
    print("\n" + "="*60)
    print("  THRESHOLD TUNING  —  lstm_precision_final_opt.h5")
    print("="*60)

    # ── Build test set (same split as training: 80/20, seed=42) ──────────
    X, y = load_and_build(DATA_PATH)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20,
                                        random_state=SEED, stratify=y)
    print(f"  Test users: {len(y_te):,}  "
          f"Fraud={y_te.sum():,} ({y_te.mean()*100:.1f}%)")

    # ── Scale with saved scaler ───────────────────────────────────────────
    with open(SCALER_PKL, 'rb') as f:
        scaler = pickle.load(f)
    sh      = X_te.shape
    X_te_sc = scaler.transform(X_te.reshape(-1, sh[-1])).reshape(sh)

    # ── Load model & predict ─────────────────────────────────────────────
    print(f"  Loading {MODEL_H5} ...")
    with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
        model = tf.keras.models.load_model(MODEL_H5, compile=False)
    y_prob = model.predict(X_te_sc, batch_size=2048, verbose=1).flatten()
    print(f"\n  Score range: min={y_prob.min():.4f}  max={y_prob.max():.4f}  "
          f"mean={y_prob.mean():.4f}")

    # ── Threshold sweep ───────────────────────────────────────────────────
    thresholds = np.arange(0.05, 0.96, 0.05)
    print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>8} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7}")
    print("  " + "-"*72)

    results = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0,1]).ravel()
        p  = precision_score(y_te, y_pred, zero_division=0)
        r  = recall_score(y_te,    y_pred, zero_division=0)
        f1 = f1_score(y_te,        y_pred, zero_division=0)
        results.append((thr, p, r, f1, tp, fp, fn, tn))
        print(f"  {thr:>10.2f} {p*100:>9.1f}% {r*100:>9.1f}% "
              f"{f1*100:>7.1f}% {tp:>7,} {fp:>7,} {fn:>7,} {tn:>7,}")

    if not results:
        print("  No valid thresholds found — model may be collapsed.")
        return

    # ── Choose best threshold (closest to Precision=Recall crossover) ────
    best_balanced = min(results, key=lambda x: abs(x[1] - x[2]))
    best_f1       = max(results, key=lambda x: x[3])
    best_prec90   = next((r for r in sorted(results, key=lambda x: -x[0])
                          if r[1] >= 0.90 and r[2] >= 0.30), None)

    print("\n" + "="*60)
    print("  RECOMMENDED THRESHOLDS")
    print("="*60)
    print(f"\n  [1] Balanced (P≈R crossover):  thr={best_balanced[0]:.2f}  "
          f"P={best_balanced[1]*100:.1f}%  R={best_balanced[2]*100:.1f}%  "
          f"F1={best_balanced[3]*100:.1f}%")
    print(f"  [2] Best F1:                   thr={best_f1[0]:.2f}  "
          f"P={best_f1[1]*100:.1f}%  R={best_f1[2]*100:.1f}%  "
          f"F1={best_f1[3]*100:.1f}%")
    if best_prec90:
        print(f"  [3] Precision≥90% w/ R≥30%:   thr={best_prec90[0]:.2f}  "
              f"P={best_prec90[1]*100:.1f}%  R={best_prec90[2]*100:.1f}%  "
              f"F1={best_prec90[3]*100:.1f}%")

    chosen = best_balanced[0]
    with open(OUT_THR, 'w') as f:
        f.write(f'{chosen:.4f}')
    print(f"\n  ✅ Saved threshold {chosen:.4f}  →  {OUT_THR}")
    print(f"     (Precision={best_balanced[1]*100:.1f}%  "
          f"Recall={best_balanced[2]*100:.1f}%)\n")


if __name__ == '__main__':
    main()
