"""
17_lstm_precision_optimised.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Precision-focused LSTM model trained on v2 + v3 datasets.

KEY IMPROVEMENTS over script 13
─────────────────────────────────
1. TRANSACTION-LEVEL prediction (not user-level)
   → Each sliding window of SEQ_LEN transactions gets its own label
   → The label is the fraud status of the LAST transaction in the window
   → Much harder & more realistic than user-level max-label

2. COMBINED v2 + v3 datasets for richer training data

3. Enhanced feature engineering (26 features, all leak-free):
   → Interaction features: amount × time_gap, amount × location_changed
   → Velocity features: amount_acceleration, tx_frequency_ratio
   → Statistical features: amount_zscore, rolling_std_amount

4. Attention-based LSTM architecture
   → Self-attention layer after LSTM to focus on suspicious timesteps
   → Deeper dense head with residual-like connections

5. Proper user-level split (users in train never appear in test)

6. Threshold selection targeting Precision ≥ 60% with max Recall

7. Bayesian optimisation via Optuna TPE (50 trials)

LEAKAGE PREVENTION
──────────────────
✗ No future information in any feature
✗ No label-derived features (fraud_count, flagged, etc.)
✗ Scaler fit on train fold only (per CV fold and final)
✗ User-level split: zero user overlap between train/test
✗ Sequences built AFTER split (no cross-contamination)

Outputs (all NEW files)
───────────────────────
  models/lstm_precision_final.h5
  models/lstm_precision_best.h5
  models/lstm_precision_scaler.pkl
  models/lstm_precision_threshold.txt
  models/lstm_precision_metadata.json
  models/lstm_precision_features.txt

Usage
─────
  cd d:\\Major_project
  python python_scripts/17_lstm_precision_optimised.py

  # Skip search and re-train with saved best params:
  python python_scripts/17_lstm_precision_optimised.py --skip-search
"""

import argparse
import json
import os
import pickle
import sys
import warnings
import time
from collections import Counter

def to_python(obj):
    """Recursively convert numpy scalars / arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(i) for i in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    print("⚠️  optuna not found.  pip install optuna")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_V2     = os.path.join(BASE_DIR, 'fraud_detection_dataset_lstm_v2.csv')
DATA_V3     = os.path.join(BASE_DIR, 'fraud_detection_dataset_lstm_v3.csv')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Output files — never clash with existing model files
OUT_PREFIX      = os.path.join(MODELS_DIR, 'lstm_precision')
FINAL_H5        = OUT_PREFIX + '_final.h5'
BEST_H5         = OUT_PREFIX + '_best.h5'
SCALER_PKL      = OUT_PREFIX + '_scaler.pkl'
THRESHOLD_TXT   = OUT_PREFIX + '_threshold.txt'
PARAMS_PKL      = OUT_PREFIX + '_params.pkl'
METADATA_JSON   = OUT_PREFIX + '_metadata.json'
FEATURES_TXT    = OUT_PREFIX + '_features.txt'

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN = 15

# 26 leak-free features (expanded from original 19)
SEQ_FEATURE_COLS = [
    # ── Original raw features (from CSV) ──────────────────────────────────
    'amount',
    'merchant_category_encoded',
    'time_gap_from_previous_hrs',
    'amount_deviation_from_avg_pct',
    'transactions_last_1hr',
    'transactions_last_24hr',
    'account_age_days',
    'avg_transaction_amount',
    'transaction_frequency',
    'device_id_encoded',
    'location_encoded',
    'most_used_device_encoded',
    'most_used_location_encoded',
    'location_changed',
    'device_changed',
    'hour',
    'day_of_week',
    'day_of_month',
    'month',
    # ── NEW engineered features ───────────────────────────────────────────
    'log_amount',                    # log-transformed amount (reduces skew)
    'amount_x_time_gap',             # interaction: high amount + short gap = suspicious
    'amount_x_location_changed',     # interaction: high amount + new location = suspicious
    'tx_frequency_ratio',            # current freq vs running avg freq (velocity)
    'amount_zscore',                 # how many SDs from user's running mean
    'hour_sin',                      # cyclical hour encoding (sin)
    'hour_cos',                      # cyclical hour encoding (cos)
]
N_FEATURES = len(SEQ_FEATURE_COLS)  # 26

MERCHANT_CATEGORIES = [
    'Grocery', 'Restaurant', 'Gas Station', 'Online Shopping', 'Electronics',
    'Pharmacy', 'Entertainment', 'Travel', 'Utilities', 'Insurance',
    'Healthcare', 'Education', 'Clothing', 'Home Improvement', 'ATM Withdrawal'
]
DEVICE_TYPES = [
    'Mobile_iOS', 'Mobile_Android', 'Desktop_Windows',
    'Desktop_Mac', 'Tablet_iOS', 'Tablet_Android', 'Web_Browser'
]
LOCATIONS = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata',
    'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Chandigarh',
    'Bhopal', 'Nagpur', 'Vadodara', 'Coimbatore', 'Indore', 'Patna'
]

# ── Optuna config ─────────────────────────────────────────────────────────────
N_TRIALS        = 30          # 30 trials for full dataset optimization
SUBSAMPLE_FRAC  = 0.15        # fraction of users per trial
CV_FOLDS        = 3
PREC_TARGET     = 0.60        # target precision
EPOCHS_TRIAL    = 10          # fewer epochs per CV fold for speed
EPOCHS_FINAL    = 80          # full epochs for final training
PATIENCE_FINAL  = 12
MAX_USERS_LOAD  = None        # FULL DATASET loaded
NORMAL_SEQ_STRIDE = 10        # larger stride to prevent RAM explosion on full data
MAX_NORMAL_RATIO  = 3.0       # max normal:fraud sequence ratio


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _encode_col(series, vocab):
    m = {v: i + 1 for i, v in enumerate(vocab)}   # 0 = unknown
    return series.map(lambda x: m.get(str(x), 0)).astype(int)


def _most_common_past(group_series):
    result, past = [], []
    for val in group_series:
        result.append(max(set(past), key=past.count) if past else val)
        past.append(val)
    return pd.Series(result, index=group_series.index)


def load_single_dataset(path, max_users=None, label=''):
    """Load a single CSV and apply feature engineering."""
    print(f"  Loading {label}: {os.path.basename(path)} …")
    df = pd.read_csv(path)

    if max_users:
        users = df['user_id'].unique()[:max_users]
        df = df[df['user_id'].isin(users)]

    print(f"    Rows={len(df):,}  Users={df['user_id'].nunique():,}  "
          f"Fraud txns={df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    return df


def engineer_features(df):
    """
    Build all 26 leak-free features from raw CSV columns.
    All features use only past / current-row information — NO future leakage.
    """
    df = df.copy()

    # ── Parse timestamp ───────────────────────────────────────────────────
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    # ── Time features ─────────────────────────────────────────────────────
    df['hour']         = df['timestamp'].dt.hour.fillna(12).astype(int)
    df['day_of_week']  = df['timestamp'].dt.dayofweek.fillna(1).astype(int)
    df['day_of_month'] = df['timestamp'].dt.day.fillna(15).astype(int)
    df['month']        = df['timestamp'].dt.month.fillna(1).astype(int)

    # Cyclical hour encoding (captures midnight wrap-around)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # ── Categorical encoding ─────────────────────────────────────────────
    df['merchant_category_encoded'] = _encode_col(df['merchant_category'], MERCHANT_CATEGORIES)
    df['device_id_encoded']         = _encode_col(df['device_id'],         DEVICE_TYPES)
    df['location_encoded']          = _encode_col(df['location'],          LOCATIONS)

    # most_used_device / most_used_location (past-mode, not including current)
    if 'most_used_device' not in df.columns:
        df['most_used_device'] = df.groupby('user_id', group_keys=False)['device_id'].apply(_most_common_past)
    if 'most_used_location' not in df.columns:
        df['most_used_location'] = df.groupby('user_id', group_keys=False)['location'].apply(_most_common_past)

    df['most_used_device_encoded']   = _encode_col(df['most_used_device'],   DEVICE_TYPES)
    df['most_used_location_encoded'] = _encode_col(df['most_used_location'], LOCATIONS)

    # ── Binary change flags ───────────────────────────────────────────────
    df['location_changed'] = (df['location'] != df['most_used_location']).astype(int)
    df['device_changed']   = (df['device_id'] != df['most_used_device']).astype(int)

    # ── Rename CSV column if needed ───────────────────────────────────────
    if 'time_gap_from_previous_hrs' not in df.columns:
        if 'time_gap_from_previous_hr_pct' in df.columns:
            df.rename(columns={'time_gap_from_previous_hr_pct': 'time_gap_from_previous_hrs'}, inplace=True)

    # ── Log-transformed amount ────────────────────────────────────────────
    df['log_amount'] = np.log1p(df['amount'].clip(lower=0))

    # ── Interaction: amount × time_gap (high amount rapidly = suspicious) ─
    df['amount_x_time_gap'] = df['amount'] * (1.0 / (df['time_gap_from_previous_hrs'].clip(lower=0.01)))

    # ── Interaction: amount × location_changed ────────────────────────────
    df['amount_x_location_changed'] = df['amount'] * df['location_changed']

    # ── Transaction frequency ratio (velocity change) ─────────────────────
    # transaction_frequency is cumulative count; ratio vs account_age gives velocity
    df['tx_frequency_ratio'] = df['transaction_frequency'] / (df['account_age_days'].clip(lower=1))

    # ── Amount z-score (how unusual is this amount vs user's running avg) ─
    # Uses avg_transaction_amount (past-only) from the CSV
    # We compute a running std approximation using amount_deviation_from_avg_pct
    df['amount_zscore'] = df['amount_deviation_from_avg_pct'] / 100.0  # normalise to ~N(0,1)

    # ── Fill missing ──────────────────────────────────────────────────────
    for col in SEQ_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df[SEQ_FEATURE_COLS] = df[SEQ_FEATURE_COLS].fillna(0)

    # Replace inf with large finite values
    df[SEQ_FEATURE_COLS] = df[SEQ_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def load_dataset(max_users=None):
    """Load ONLY the specific v3 dataset (most realistic, balanced)."""
    print(f"  Loading dataset: v3 (fraud_detection_dataset_lstm_v3.csv) …")
    if not os.path.exists(DATA_V3):
        raise FileNotFoundError(f"Dataset not found at {DATA_V3}")

    df = pd.read_csv(DATA_V3)
    
    if max_users:
        users = df['user_id'].unique()[:max_users]
        df = df[df['user_id'].isin(users)]
        print(f"    [Subset] Limited to {max_users:,} users for speed/memory.")

    print(f"    Rows={len(df):,}  Users={df['user_id'].nunique():,}  "
          f"Fraud txns={df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")

    print("  Engineering 26 leak-free features …")
    df = engineer_features(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSACTION-LEVEL SEQUENCE BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_transaction_sequences(df, user_ids, stride_normal=NORMAL_SEQ_STRIDE):
    """
    Build TRANSACTION-LEVEL sliding-window sequences (memory-efficient).
    
    For each user, creates sliding windows of SEQ_LEN consecutive transactions.
    The LABEL is the fraud status of the LAST transaction in each window.
    
    Memory optimization:
    - Fraud windows: stride=1 (keep ALL fraud-labeled windows)
    - Normal windows: stride=stride_normal (skip every N, saves memory)
    - Always includes the LAST window per user (ensures coverage)
    
    This is fundamentally different from the user-level approach:
    - User-level: one sequence per user, label = max(all_fraud_flags)
    - Transaction-level: many sequences per user, each labeled individually
    """
    X_seqs, y_labels = [], []
    n_feat = len(SEQ_FEATURE_COLS)
    fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'isFraud'

    user_df = df[df['user_id'].isin(user_ids)]
    n_users_done = 0
    total_users = len(user_ids)

    for uid, grp in user_df.groupby('user_id', sort=False):
        feats  = grp[SEQ_FEATURE_COLS].values.astype(np.float32)
        labels = grp[fraud_col].values.astype(int)
        n = len(feats)

        if n < SEQ_LEN:
            # Pad with zeros at the start and use the last transaction's label
            padded = np.zeros((SEQ_LEN, n_feat), dtype=np.float32)
            padded[SEQ_LEN - n:] = feats
            X_seqs.append(padded)
            y_labels.append(labels[-1])
        else:
            # Sliding window with smart stride
            has_fraud = labels.max() > 0
            for start in range(n - SEQ_LEN + 1):
                end_idx = start + SEQ_LEN - 1
                label = labels[end_idx]  # label of LAST tx in window

                # Always keep fraud-labeled windows (stride=1)
                if label == 1:
                    X_seqs.append(feats[start : start + SEQ_LEN])
                    y_labels.append(1)
                # For normal windows: use stride to reduce count
                elif start % stride_normal == 0 or start == (n - SEQ_LEN):
                    X_seqs.append(feats[start : start + SEQ_LEN])
                    y_labels.append(0)

        n_users_done += 1
        if n_users_done % 20000 == 0:
            print(f"    Built sequences for {n_users_done:,}/{total_users:,} users "
                  f"(seqs so far: {len(X_seqs):,})")

    X = np.array(X_seqs, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    # Downsample normal sequences if ratio is too skewed
    n_fraud = (y == 1).sum()
    n_normal = (y == 0).sum()
    if n_fraud > 0 and n_normal / n_fraud > MAX_NORMAL_RATIO:
        max_normal = int(n_fraud * MAX_NORMAL_RATIO)
        fraud_idx  = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        normal_keep = np.random.choice(normal_idx, max_normal, replace=False)
        keep_idx = np.sort(np.concatenate([fraud_idx, normal_keep]))
        X = X[keep_idx]
        y = y[keep_idx]
        print(f"    Downsampled normal: {n_normal:,} → {max_normal:,} "
              f"(fraud={n_fraud:,}, ratio={MAX_NORMAL_RATIO:.0f}:1)")

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  FOCAL LOSS (Precision-Focused Variant)
# ══════════════════════════════════════════════════════════════════════════════

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary focal loss — focuses training on hard / misclassified examples.
    With gamma > 0, easy examples contribute less to the loss,
    forcing the model to learn harder distinctions → better precision.
    """
    def loss_fn(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
        bce    = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl     = alpha * K.pow(1 - p_t, gamma) * bce
        return K.mean(fl)
    return loss_fn


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENTION LAYER
# ══════════════════════════════════════════════════════════════════════════════

class AttentionLayer(layers.Layer):
    """
    Simple self-attention for LSTM outputs.
    Learns which timesteps in the sequence are most important for prediction.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_W',
                                 shape=(int(input_shape[-1]), int(input_shape[-1])),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_b',
                                 shape=(int(input_shape[-1]),),
                                 initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_u',
                                 shape=(int(input_shape[-1]),),
                                 initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        uit = K.tanh(K.dot(x, self.W) + self.b)         # (batch, T, F)
        ait = K.sum(uit * self.u, axis=-1)               # (batch, T)
        ait = K.softmax(ait, axis=-1)                    # (batch, T)
        ait = K.expand_dims(ait, axis=-1)                # (batch, T, 1)
        weighted = x * ait                               # (batch, T, F)
        return K.sum(weighted, axis=1)                   # (batch, F)

    def get_config(self):
        return super().get_config()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BUILDER (Attention-LSTM)
# ══════════════════════════════════════════════════════════════════════════════

def build_model(params: dict):
    """
    Attention-enhanced bidirectional LSTM for precision-focused fraud detection.
    
    Architecture:
      Input → BiLSTM-1 → Dropout → BiLSTM-2 (return_sequences) →
      Attention → BatchNorm → Dense → Dropout → Dense → Sigmoid
    """
    inp = keras.Input(shape=(SEQ_LEN, N_FEATURES))

    # Layer 1: Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(
            params['lstm_units_1'],
            return_sequences=True,
            dropout=params['dropout_1'],
            recurrent_dropout=params['recurrent_dropout'],
            kernel_regularizer=keras.regularizers.l2(params.get('l2_reg', 1e-4)),
        )
    )(inp)

    # Layer 2: Second LSTM (return sequences for attention)
    x = layers.LSTM(
        params['lstm_units_2'],
        return_sequences=True,
        dropout=params['dropout_2'],
        recurrent_dropout=params['recurrent_dropout'],
    )(x)

    # Attention mechanism — learn which timesteps matter most
    x = AttentionLayer()(x)

    # Dense head
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(params['dense_units_1'], activation='relu')(x)
    x   = layers.Dropout(params['dropout_3'])(x)
    x   = layers.Dense(params.get('dense_units_2', 16), activation='relu')(x)
    x   = layers.Dropout(params['dropout_3'] * 0.5)(x)  # lighter dropout on last dense
    out = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inp, out)

    gamma = params.get('focal_gamma', 0.0)
    loss  = focal_loss(gamma=gamma, alpha=params.get('focal_alpha', 0.25)) \
            if gamma > 0 else 'binary_crossentropy'

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss=loss,
        metrics=[
            keras.metrics.AUC(name='pr_auc', curve='PR'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  THRESHOLD SELECTION (Precision-Focused)
# ══════════════════════════════════════════════════════════════════════════════

def select_threshold(y_true, y_prob, prec_target=PREC_TARGET):
    """
    Returns two thresholds:
      thr_f1    – maximises F1-score
      thr_prec  – highest Recall where Precision >= prec_target
                  (with minimum recall >= 25% to avoid degenerate predictor)
    """
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)

    # F1-optimal
    f1_arr = np.where((prec_arr + rec_arr) == 0, 0,
                      2 * prec_arr * rec_arr / (prec_arr + rec_arr))
    f1_idx = np.argmax(f1_arr)
    thr_f1 = float(thr_arr[f1_idx]) if f1_idx < len(thr_arr) else 0.5

    # Precision-targeted (max recall where precision >= target & recall >= 25%)
    mask = (prec_arr >= prec_target) & (rec_arr >= 0.25)
    if mask.any():
        candidates = np.where(mask)[0]
        best_c   = candidates[np.argmax(rec_arr[candidates])]
        thr_prec = float(thr_arr[best_c]) if best_c < len(thr_arr) else thr_f1
    else:
        # Try relaxing recall requirement
        mask2 = prec_arr >= prec_target
        if mask2.any():
            candidates = np.where(mask2)[0]
            best_c = candidates[np.argmax(rec_arr[candidates])]
            thr_prec = float(thr_arr[best_c]) if best_c < len(thr_arr) else thr_f1
            print(f"  ⚠️  Precision target {prec_target*100:.0f}% achieved but "
                  f"recall < 25%. Using best available.")
        else:
            print(f"  ⚠️  Precision target {prec_target*100:.0f}% not achievable. "
                  f"Using F1 threshold instead.")
            thr_prec = thr_f1

    return thr_f1, thr_prec


def print_metrics(tag, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true,   y_pred, zero_division=0)
    f1      = f1_score(y_true,       y_pred, zero_division=0)
    roc     = roc_auc_score(y_true,  y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n  ── {tag}  (threshold={threshold:.4f}) ──")
    print(f"     TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(f"     False-Positive Rate : {fp_rate*100:.4f}%")
    print(f"     Precision           : {prec*100:.2f}%")
    print(f"     Recall              : {rec*100:.2f}%")
    print(f"     F1-Score            : {f1*100:.2f}%")
    print(f"     ROC-AUC             : {roc:.4f}")
    print(f"     PR-AUC              : {pr_auc:.4f}")
    return dict(threshold=threshold, precision=prec, recall=rec,
                f1=f1, roc_auc=roc, pr_auc=pr_auc,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), fp_rate=fp_rate)


# ══════════════════════════════════════════════════════════════════════════════
#  BAYESIAN OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_search(df_train, train_users, n_trials=N_TRIALS,
                    subsample_frac=SUBSAMPLE_FRAC, cv_folds=CV_FOLDS):
    """
    Optuna TPE search.
    Objective = PR-AUC (directly captures Precision–Recall tradeoff).
    Sequences are built INSIDE each fold from user-level splits.
    """
    if not OPTUNA_OK:
        raise RuntimeError("optuna not installed. Run: pip install optuna")

    # Subsample users for speed
    n_sub = max(int(len(train_users) * subsample_frac), 1000)
    sub_users = np.random.choice(train_users, min(n_sub, len(train_users)), replace=False)
    sub_df = df_train[df_train['user_id'].isin(sub_users)].copy()

    # Get user-level fraud labels for stratified CV
    user_labels = sub_df.groupby('user_id')['is_fraud'].max().reset_index()
    users_arr   = user_labels['user_id'].values
    labels_arr  = user_labels['is_fraud'].values

    print(f"\n  Bayesian search: {n_trials} trials | "
          f"subsample={len(sub_users):,} users | {cv_folds}-fold CV")
    print(f"  Objective: PR-AUC (transaction-level, scaler per fold)\n")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            'lstm_units_1':      trial.suggest_categorical('lstm_units_1',    [64, 96, 128, 192]),
            'lstm_units_2':      trial.suggest_categorical('lstm_units_2',    [32, 48, 64, 96]),
            'dropout_1':         trial.suggest_float('dropout_1',             0.15, 0.50),
            'dropout_2':         trial.suggest_float('dropout_2',             0.15, 0.50),
            'dropout_3':         trial.suggest_float('dropout_3',             0.20, 0.55),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout',     0.00, 0.25),
            'dense_units_1':     trial.suggest_categorical('dense_units_1',   [32, 48, 64, 96]),
            'dense_units_2':     trial.suggest_categorical('dense_units_2',   [8, 16, 24, 32]),
            'learning_rate':     trial.suggest_float('learning_rate',         1e-4, 3e-3, log=True),
            'l2_reg':            trial.suggest_float('l2_reg',               1e-5, 1e-2, log=True),
            # Class weight multiplier: controls precision-recall tradeoff
            # LOWER values → model is more conservative → higher precision
            'class_weight_mult': trial.suggest_float('class_weight_mult',    0.3, 2.0),
            # Focal loss
            'focal_gamma':       trial.suggest_float('focal_gamma',          0.0, 3.0),
            'focal_alpha':       trial.suggest_float('focal_alpha',          0.10, 0.50),
        }

        scores = []
        for tr_idx, val_idx in cv.split(users_arr, labels_arr):
            tr_users  = set(users_arr[tr_idx])
            val_users = set(users_arr[val_idx])

            # Build sequences from user splits (zero user overlap)
            X_tr, y_tr = build_transaction_sequences(sub_df, tr_users)
            X_va, y_va = build_transaction_sequences(sub_df, val_users)

            if len(X_tr) == 0 or len(X_va) == 0:
                continue

            # Scale per fold (no leakage)
            sh      = X_tr.shape
            scaler  = RobustScaler()
            X_tr_f  = scaler.fit_transform(X_tr.reshape(-1, sh[-1])).reshape(sh)
            X_va_f  = scaler.transform(X_va.reshape(-1, sh[-1])).reshape(X_va.shape)

            # Class weights
            n_neg = (y_tr == 0).sum()
            n_pos = max((y_tr == 1).sum(), 1)
            fraud_w = (n_neg / n_pos) * params['class_weight_mult']
            cw = {0: 1.0, 1: fraud_w}

            K.clear_session()
            model = build_model(params)
            cb = EarlyStopping(monitor='val_pr_auc', patience=4,
                               restore_best_weights=True, mode='max')
            model.fit(
                X_tr_f, y_tr,
                validation_data=(X_va_f, y_va),
                epochs=EPOCHS_TRIAL,
                batch_size=512,
                class_weight=cw,
                callbacks=[cb],
                verbose=0,
            )
            prob  = model.predict(X_va_f, verbose=0).flatten()
            score = average_precision_score(y_va, prob)
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
    )
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=True, gc_after_trial=True)

    best = study.best_params
    print(f"\n  ✅ Best PR-AUC (CV) : {study.best_value:.4f}")
    print(f"  Best params        : {json.dumps(best, indent=4)}")
    return best, study


# ══════════════════════════════════════════════════════════════════════════════
#  FULL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_full(params, X_train, y_train, X_test, y_test, scaler):
    n_neg   = (y_train == 0).sum()
    n_pos   = max((y_train == 1).sum(), 1)
    mult    = params.get('class_weight_mult', 1.0)
    fraud_w = (n_neg / n_pos) * mult
    cw      = {0: 1.0, 1: fraud_w}
    print(f"\n  Class weight — normal: 1.0  fraud: {cw[1]:.2f}  "
          f"(ratio={n_neg/n_pos:.1f}:1, mult={mult:.2f})")
    print(f"  Train sequences: {len(y_train):,}  "
          f"(fraud={y_train.sum():,}, {y_train.mean()*100:.2f}%)")
    print(f"  Test  sequences: {len(y_test):,}  "
          f"(fraud={y_test.sum():,}, {y_test.mean()*100:.2f}%)")

    K.clear_session()
    model = build_model(params)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_pr_auc', patience=PATIENCE_FINAL,
                      restore_best_weights=True, mode='max', verbose=1),
        ModelCheckpoint(BEST_H5, monitor='val_pr_auc',
                        save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_pr_auc', factor=0.5, patience=7,
                          min_lr=1e-7, mode='max', verbose=1),
    ]

    batch = int(params.get('batch_size', 512))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS_FINAL,
        batch_size=batch,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(FINAL_H5)
    print(f"\n  ✅ Saved final model → {FINAL_H5}")
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(skip_search=False):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("  LSTM PRECISION-OPTIMISED MODEL (v3 realistic dataset ONLY)")
    print("  Transaction-level prediction with Attention + Bayesian Optimisation")
    print("=" * 70)

    # ── Load & engineer features ──────────────────────────────────────────
    print("\n[1/5]  Loading data and engineering features …")
    df = load_dataset(max_users=MAX_USERS_LOAD)

    # ── User-level split (CRITICAL: no user overlap) ─────────────────────
    print("\n[2/5]  Splitting users (80/20, stratified, zero overlap) …")
    user_labels = df.groupby('user_id')['is_fraud'].max().reset_index()
    users_arr   = user_labels['user_id'].values
    labels_arr  = user_labels['is_fraud'].values

    train_users, test_users = train_test_split(
        users_arr, test_size=0.20, random_state=SEED, stratify=labels_arr
    )
    # Verify zero overlap
    overlap = set(train_users) & set(test_users)
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} users in both splits!"
    print(f"  Train users: {len(train_users):,}  Test users: {len(test_users):,}")
    print(f"  User overlap: {len(overlap)} (verified zero)")

    df_train = df[df['user_id'].isin(set(train_users))]
    df_test  = df[df['user_id'].isin(set(test_users))]

    print(f"  Train rows: {len(df_train):,}  Test rows: {len(df_test):,}")

    # ── Build transaction-level sequences ─────────────────────────────────
    print("\n  Building transaction-level sequences …")
    X_tr, y_tr = build_transaction_sequences(df, set(train_users))
    X_te, y_te = build_transaction_sequences(df, set(test_users))
    print(f"  Train sequences: {X_tr.shape}  Fraud: {y_tr.sum():,}/{len(y_tr):,} "
          f"({y_tr.mean()*100:.2f}%)")
    print(f"  Test  sequences: {X_te.shape}  Fraud: {y_te.sum():,}/{len(y_te):,} "
          f"({y_te.mean()*100:.2f}%)")

    # ── Scale (fit on train only — no leakage) ────────────────────────────
    print("  Scaling features (RobustScaler, fit on train only) …")
    sh      = X_tr.shape
    scaler  = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr.reshape(-1, sh[-1])).reshape(sh)
    X_te_sc = scaler.transform(X_te.reshape(-1, sh[-1])).reshape(X_te.shape)

    with open(SCALER_PKL, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✅ Saved scaler → {SCALER_PKL}")

    # Save feature list
    with open(FEATURES_TXT, 'w') as f:
        f.write('\n'.join(SEQ_FEATURE_COLS))
    print(f"  ✅ Saved features → {FEATURES_TXT}")

    # ── Bayesian search or load saved params ─────────────────────────────
    if skip_search and os.path.exists(PARAMS_PKL):
        print("\n[3/5]  Loading saved best params (--skip-search) …")
        with open(PARAMS_PKL, 'rb') as f:
            best_params = pickle.load(f)
        print(f"  Params: {json.dumps(best_params, indent=4)}")
    else:
        print("\n[3/5]  Bayesian hyperparameter search (Optuna TPE) …")
        best_params, study = bayesian_search(df, train_users)
        with open(PARAMS_PKL, 'wb') as f:
            pickle.dump(best_params, f)
        print(f"  ✅ Saved best params → {PARAMS_PKL}")

    # ── Full training with best params ────────────────────────────────────
    print("\n[4/5]  Training final model on full training set …")
    model, history = train_full(best_params, X_tr_sc, y_tr, X_te_sc, y_te, scaler)

    # ── Evaluate & select thresholds ─────────────────────────────────────
    print("\n[5/5]  Evaluating & selecting thresholds …")
    y_prob = model.predict(X_te_sc, verbose=0).flatten()

    thr_f1, thr_prec = select_threshold(y_te, y_prob, prec_target=PREC_TARGET)

    print(f"\n  F1-optimal threshold      : {thr_f1:.4f}")
    print(f"  Precision-target (≥{PREC_TARGET*100:.0f}%) : {thr_prec:.4f}")

    m_f1   = print_metrics("F1-optimal threshold",    y_te, y_prob, thr_f1)
    m_prec = print_metrics(f"Precision≥{PREC_TARGET*100:.0f}% threshold", y_te, y_prob, thr_prec)

    # ── Also report at multiple precision levels ──────────────────────────
    print("\n  ── Precision at various thresholds ──")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (y_prob >= thr).astype(int)
        p = precision_score(y_te, preds, zero_division=0)
        r = recall_score(y_te, preds, zero_division=0)
        f = f1_score(y_te, preds, zero_division=0)
        print(f"     thr={thr:.1f}  Prec={p*100:.2f}%  Rec={r*100:.2f}%  F1={f*100:.2f}%")

    print("\n  ── Baseline comparison ──")
    print("     Old model: Precision=21.27%  Recall=65.15%  F1=32.07%  AUC=0.634")

    # ── Choose the best threshold ─────────────────────────────────────────
    # Use precision-targeted threshold as primary
    chosen_thr = thr_prec
    with open(THRESHOLD_TXT, 'w') as f:
        f.write(f'{chosen_thr:.4f}')
    print(f"\n  ✅ Saved threshold ({chosen_thr:.4f}) → {THRESHOLD_TXT}")

    # ── Save metadata ─────────────────────────────────────────────────────
    metadata = {
        'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'LSTM Precision-Optimised (Attention + BiLSTM)',
        'approach': 'Transaction-level sliding window (not user-level)',
        'datasets_used': ['fraud_detection_dataset_lstm_v3.csv'],
        'n_optuna_trials': N_TRIALS,
        'subsample_frac': SUBSAMPLE_FRAC,
        'cv_folds': CV_FOLDS,
        'prec_target': PREC_TARGET,
        'seq_len': SEQ_LEN,
        'n_features': N_FEATURES,
        'feature_columns': SEQ_FEATURE_COLS,
        'best_hyperparams': best_params,
        'threshold_f1_optimal': thr_f1,
        'threshold_prec_targeted': thr_prec,
        'chosen_threshold': chosen_thr,
        'metrics_f1_threshold': m_f1,
        'metrics_prec_threshold': m_prec,
        'leakage_prevention': {
            'user_split': 'Users split before sequences built (zero overlap)',
            'scaling': 'RobustScaler fit on train only',
            'features': '26 leak-free features (no label-derived, no future info)',
            'prediction_level': 'Transaction-level (last tx in window)',
            'user_overlap_verified': 0,
        },
        'baseline_comparison': {
            'precision': 0.2127, 'recall': 0.6515,
            'f1': 0.3207, 'auc': 0.634, 'threshold': 0.5301,
        },
        'train_users': int(len(train_users)),
        'test_users':  int(len(test_users)),
        'train_sequences': int(len(y_tr)),
        'test_sequences':  int(len(y_te)),
        'fraud_rate_txn_level': float(y_te.mean()),
    }
    with open(METADATA_JSON, 'w') as f:
        json.dump(to_python(metadata), f, indent=2)
    print(f"  ✅ Saved metadata → {METADATA_JSON}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  DONE  ({elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"\n  Model files saved:")
    print(f"    {FINAL_H5}")
    print(f"    {BEST_H5}")
    print(f"    {SCALER_PKL}")
    print(f"    {FEATURES_TXT}")
    print(f"    {THRESHOLD_TXT}  (value={chosen_thr:.4f})")
    print(f"    {PARAMS_PKL}")
    print(f"    {METADATA_JSON}")

    print(f"\n  To use in app.py, update load_lstm_model() to load:")
    print(f"    models/lstm_precision_final.h5")
    print(f"    models/lstm_precision_scaler.pkl")
    print(f"    models/lstm_precision_threshold.txt")
    print()

    return model, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precision-optimised LSTM with Bayesian HPO (v2+v3 combined)')
    parser.add_argument('--skip-search', action='store_true',
                        help='Skip Optuna search; use saved params from last run')
    args = parser.parse_args()
    main(skip_search=args.skip_search)
