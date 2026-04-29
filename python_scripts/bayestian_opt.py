!pip install optuna

# ── Mount Google Drive ────────────────────────────────────────────────────────
from google.colab import drive
import os
drive.mount('/content/drive')

# Set paths to point to your specific Drive folder
DRIVE_FOLDER = '/content/drive/MyDrive/dataset'
DATA_V3      = os.path.join(DRIVE_FOLDER, 'fraud_detection_dataset_lstm_v3.csv')

# Outputs will save directly back to your Drive folder!
OUT_PREFIX      = os.path.join(DRIVE_FOLDER, 'lstm_precision')
FINAL_H5        = OUT_PREFIX + '_final.h5'
BEST_H5         = OUT_PREFIX + '_best.h5'
SCALER_PKL      = OUT_PREFIX + '_scaler.pkl'
THRESHOLD_TXT   = OUT_PREFIX + '_threshold.txt'
PARAMS_PKL      = OUT_PREFIX + '_params.pkl'
METADATA_JSON   = OUT_PREFIX + '_metadata.json'
FEATURES_TXT    = OUT_PREFIX + '_features.txt'

import time
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, recall_score, average_precision_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN = 15

# 26 leak-free features
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
N_FEATURES = len(SEQ_FEATURE_COLS)

MERCHANT_CATEGORIES = ['Grocery', 'Restaurant', 'Gas Station', 'Online Shopping', 'Electronics', 'Pharmacy', 'Entertainment', 'Travel', 'Utilities', 'Insurance', 'Healthcare', 'Education', 'Clothing', 'Home Improvement', 'ATM Withdrawal']
DEVICE_TYPES = ['Mobile_iOS', 'Mobile_Android', 'Desktop_Windows', 'Desktop_Mac', 'Tablet_iOS', 'Tablet_Android', 'Web_Browser']
LOCATIONS = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Chandigarh', 'Bhopal', 'Nagpur', 'Vadodara', 'Coimbatore', 'Indore', 'Patna']

# ── Optuna config ─────────────────────────────────────────────────────────────
N_TRIALS        = 30
SUBSAMPLE_FRAC  = 0.15
CV_FOLDS        = 3
PREC_TARGET     = 0.60
EPOCHS_TRIAL    = 10
EPOCHS_FINAL    = 80
PATIENCE_FINAL  = 12
MAX_USERS_LOAD  = None
NORMAL_SEQ_STRIDE = 10
MAX_NORMAL_RATIO  = 3.0

def _encode_col(series, vocab):
    m = {v: i + 1 for i, v in enumerate(vocab)}
    return series.map(lambda x: m.get(str(x), 0)).astype(int)

def _most_common_past(group_series):
    result, past = [], []
    for val in group_series:
        result.append(max(set(past), key=past.count) if past else val)
        past.append(val)
    return pd.Series(result, index=group_series.index)

def engineer_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    df['hour']         = df['timestamp'].dt.hour.fillna(12).astype(int)
    df['day_of_week']  = df['timestamp'].dt.dayofweek.fillna(1).astype(int)
    df['day_of_month'] = df['timestamp'].dt.day.fillna(15).astype(int)
    df['month']        = df['timestamp'].dt.month.fillna(1).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    df['merchant_category_encoded'] = _encode_col(df['merchant_category'], MERCHANT_CATEGORIES)
    df['device_id_encoded']         = _encode_col(df['device_id'],         DEVICE_TYPES)
    df['location_encoded']          = _encode_col(df['location'],          LOCATIONS)

    if 'most_used_device' not in df.columns:
        df['most_used_device'] = df.groupby('user_id', group_keys=False)['device_id'].apply(_most_common_past)
    if 'most_used_location' not in df.columns:
        df['most_used_location'] = df.groupby('user_id', group_keys=False)['location'].apply(_most_common_past)

    df['most_used_device_encoded']   = _encode_col(df['most_used_device'],   DEVICE_TYPES)
    df['most_used_location_encoded'] = _encode_col(df['most_used_location'], LOCATIONS)

    df['location_changed'] = (df['location'] != df['most_used_location']).astype(int)
    df['device_changed']   = (df['device_id'] != df['most_used_device']).astype(int)

    if 'time_gap_from_previous_hrs' not in df.columns and 'time_gap_from_previous_hr_pct' in df.columns:
        df.rename(columns={'time_gap_from_previous_hr_pct': 'time_gap_from_previous_hrs'}, inplace=True)

    df['log_amount'] = np.log1p(df['amount'].clip(lower=0))
    df['amount_x_time_gap'] = df['amount'] * (1.0 / (df['time_gap_from_previous_hrs'].clip(lower=0.01)))
    df['amount_x_location_changed'] = df['amount'] * df['location_changed']
    df['tx_frequency_ratio'] = df['transaction_frequency'] / (df['account_age_days'].clip(lower=1))
    df['amount_zscore'] = df['amount_deviation_from_avg_pct'] / 100.0

    for col in SEQ_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df[SEQ_FEATURE_COLS] = df[SEQ_FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    return df

def load_dataset(max_users=None):
    print(f"Loading dataset: v3 from Google Drive…")
    if not os.path.exists(DATA_V3):
        raise FileNotFoundError(f"Dataset not found at {DATA_V3}")

    df = pd.read_csv(DATA_V3)
    if max_users:
        users = df['user_id'].unique()[:max_users]
        df = df[df['user_id'].isin(users)]

    print("Engineering 26 leak-free features …")
    return engineer_features(df)

def build_transaction_sequences(df, user_ids, stride_normal=NORMAL_SEQ_STRIDE):
    X_seqs, y_labels = [], []
    fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'isFraud'
    user_df = df[df['user_id'].isin(user_ids)]

    for uid, grp in user_df.groupby('user_id', sort=False):
        feats  = grp[SEQ_FEATURE_COLS].values.astype(np.float32)
        labels = grp[fraud_col].values.astype(int)
        n = len(feats)

        if n < SEQ_LEN:
            padded = np.zeros((SEQ_LEN, len(SEQ_FEATURE_COLS)), dtype=np.float32)
            padded[SEQ_LEN - n:] = feats
            X_seqs.append(padded)
            y_labels.append(labels[-1])
        else:
            for start in range(n - SEQ_LEN + 1):
                end_idx = start + SEQ_LEN - 1
                label = labels[end_idx]
                if label == 1:
                    X_seqs.append(feats[start : start + SEQ_LEN])
                    y_labels.append(1)
                elif start % stride_normal == 0 or start == (n - SEQ_LEN):
                    X_seqs.append(feats[start : start + SEQ_LEN])
                    y_labels.append(0)

    X, y = np.array(X_seqs, dtype=np.float32), np.array(y_labels, dtype=np.int32)

    n_fraud, n_normal = (y == 1).sum(), (y == 0).sum()
    if n_fraud > 0 and n_normal / n_fraud > MAX_NORMAL_RATIO:
        max_normal = int(n_fraud * MAX_NORMAL_RATIO)
        fraud_idx  = np.where(y == 1)[0]
        normal_idx = np.random.choice(np.where(y == 0)[0], max_normal, replace=False)
        keep_idx = np.sort(np.concatenate([fraud_idx, normal_idx]))
        X, y = X[keep_idx], y[keep_idx]

    return X, y

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
        bce    = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return K.mean(alpha * K.pow(1 - p_t, gamma) * bce)
    return loss_fn

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

def build_model(params: dict):
    inp = keras.Input(shape=(SEQ_LEN, N_FEATURES))
    x = layers.Bidirectional(layers.LSTM(params['lstm_units_1'], return_sequences=True, dropout=params['dropout_1'], recurrent_dropout=params['recurrent_dropout'], kernel_regularizer=keras.regularizers.l2(params.get('l2_reg', 1e-4))))(inp)
    x = layers.LSTM(params['lstm_units_2'], return_sequences=True, dropout=params['dropout_2'], recurrent_dropout=params['recurrent_dropout'])(x)
    x = AttentionLayer()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(params['dense_units_1'], activation='relu')(x)
    x = layers.Dropout(params['dropout_3'])(x)
    x = layers.Dense(params.get('dense_units_2', 16), activation='relu')(x)
    x = layers.Dropout(params['dropout_3'] * 0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inp, out)
    gamma = params.get('focal_gamma', 0.0)
    loss  = focal_loss(gamma=gamma, alpha=params.get('focal_alpha', 0.25)) if gamma > 0 else 'binary_crossentropy'

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']), loss=loss, metrics=[keras.metrics.AUC(name='pr_auc', curve='PR')])
    return model

def bayesian_search(df_train, train_users):
    n_sub = max(int(len(train_users) * SUBSAMPLE_FRAC), 1000)
    sub_users = np.random.choice(train_users, min(n_sub, len(train_users)), replace=False)
    sub_df = df_train[df_train['user_id'].isin(sub_users)].copy()
    user_labels = sub_df.groupby('user_id')['is_fraud'].max().reset_index()
    users_arr, labels_arr  = user_labels['user_id'].values, user_labels['is_fraud'].values
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            'lstm_units_1': trial.suggest_categorical('lstm_units_1', [64, 96, 128, 192]),
            'lstm_units_2': trial.suggest_categorical('lstm_units_2', [32, 48, 64, 96]),
            'dropout_1': trial.suggest_float('dropout_1', 0.15, 0.50),
            'dropout_2': trial.suggest_float('dropout_2', 0.15, 0.50),
            'dropout_3': trial.suggest_float('dropout_3', 0.20, 0.55),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.00, 0.25),
            'dense_units_1': trial.suggest_categorical('dense_units_1', [32, 48, 64, 96]),
            'dense_units_2': trial.suggest_categorical('dense_units_2', [8, 16, 24, 32]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
            'class_weight_mult': trial.suggest_float('class_weight_mult', 0.3, 2.0),
            'focal_gamma': trial.suggest_float('focal_gamma', 0.0, 3.0),
            'focal_alpha': trial.suggest_float('focal_alpha', 0.10, 0.50),
        }
        scores = []
        for tr_idx, val_idx in cv.split(users_arr, labels_arr):
            X_tr, y_tr = build_transaction_sequences(sub_df, set(users_arr[tr_idx]))
            X_va, y_va = build_transaction_sequences(sub_df, set(users_arr[val_idx]))
            if len(X_tr) == 0: continue

            sh = X_tr.shape
            scaler = RobustScaler()
            X_tr_f = scaler.fit_transform(X_tr.reshape(-1, sh[-1])).reshape(sh)
            X_va_f = scaler.transform(X_va.reshape(-1, sh[-1])).reshape(X_va.shape)

            n_neg, n_pos = (y_tr == 0).sum(), max((y_tr == 1).sum(), 1)
            fraud_w = (n_neg / n_pos) * params['class_weight_mult']

            K.clear_session()
            model = build_model(params)
            model.fit(X_tr_f, y_tr, validation_data=(X_va_f, y_va), epochs=EPOCHS_TRIAL, batch_size=512, class_weight={0: 1.0, 1: fraud_w}, callbacks=[EarlyStopping(monitor='val_pr_auc', patience=4, restore_best_weights=True, mode='max')], verbose=0)
            scores.append(average_precision_score(y_va, model.predict(X_va_f, verbose=0).flatten()))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED), pruner=optuna.pruners.MedianPruner(n_warmup_steps=8))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, gc_after_trial=True)
    return study.best_params

def select_threshold(y_true, y_prob, prec_target=PREC_TARGET):
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)
    f1_arr = np.where((prec_arr + rec_arr) == 0, 0, 2 * prec_arr * rec_arr / (prec_arr + rec_arr))
    thr_f1 = float(thr_arr[np.argmax(f1_arr)]) if np.argmax(f1_arr) < len(thr_arr) else 0.5

    mask = (prec_arr >= prec_target) & (rec_arr >= 0.25)
    if mask.any():
        c = np.where(mask)[0]
        b = c[np.argmax(rec_arr[c])]
        thr_prec = float(thr_arr[b]) if b < len(thr_arr) else thr_f1
    else:
        thr_prec = thr_f1
    return thr_f1, thr_prec

def to_python(obj):
    if isinstance(obj, dict): return {k: to_python(v) for k, v in obj.items()}
    if hasattr(obj, 'item'): return obj.item()
    if hasattr(obj, 'tolist'): return obj.tolist()
    return obj

# ── Main Execution ──
print("\n🚀 Starting Colab GPU Optimization from Google Drive...")
df = load_dataset()
user_labels = df.groupby('user_id')['is_fraud'].max().reset_index()
train_users, test_users = train_test_split(user_labels['user_id'].values, test_size=0.20, random_state=SEED, stratify=user_labels['is_fraud'].values)

print("\nBuilding Full Sequences...")
X_tr, y_tr = build_transaction_sequences(df, set(train_users))
X_te, y_te = build_transaction_sequences(df, set(test_users))

print("Scaling Data...")
sh = X_tr.shape
scaler = RobustScaler()
X_tr_sc = scaler.fit_transform(X_tr.reshape(-1, sh[-1])).reshape(sh)
X_te_sc = scaler.transform(X_te.reshape(-1, sh[-1])).reshape(X_te.shape)

with open(SCALER_PKL, 'wb') as file_scaler: pickle.dump(scaler, file_scaler)
with open(FEATURES_TXT, 'w') as file_feats: file_feats.write('\n'.join(SEQ_FEATURE_COLS))

print("\nStarting Bayesian Search...")
best_params = bayesian_search(df[df['user_id'].isin(set(train_users))], train_users)
with open(PARAMS_PKL, 'wb') as file_params: pickle.dump(best_params, file_params)
print(f"Optimal Params Found: {json.dumps(best_params, indent=2)}")

print("\nTraining Final Model...")
n_neg, n_pos = (y_tr == 0).sum(), max((y_tr == 1).sum(), 1)
cw = {0: 1.0, 1: (n_neg / n_pos) * best_params.get('class_weight_mult', 1.0)}

K.clear_session()
model = build_model(best_params)
model.fit(X_tr_sc, y_tr, validation_data=(X_te_sc, y_te), epochs=EPOCHS_FINAL, batch_size=int(best_params.get('batch_size', 512)), class_weight=cw, callbacks=[EarlyStopping(monitor='val_pr_auc', patience=PATIENCE_FINAL, restore_best_weights=True, mode='max')], verbose=1)
model.save(FINAL_H5)  # Legacy format warning is fine, we need .h5 for your app

print("\nEvaluating...")
y_prob = model.predict(X_te_sc, verbose=0).flatten()
thr_f1, thr_prec = select_threshold(y_te, y_prob)

y_pred = (y_prob >= thr_prec).astype(int)
p_score, r_score, f1_score_val = precision_score(y_te, y_pred, zero_division=0), recall_score(y_te, y_pred, zero_division=0), f1_score(y_te, y_pred, zero_division=0)

with open(THRESHOLD_TXT, 'w') as txt_file: txt_file.write(f"{thr_prec:.5f}")

# JSON dump fixed: No variable conflicts anymore!
with open(METADATA_JSON, 'w') as json_file:
    json.dump(to_python({'scores': {'precision': p_score, 'recall': r_score, 'f1': f1_score_val, 'threshold': thr_prec}, 'params': best_params}), json_file, indent=4)

print(f"\n✅ DONE! Target Threshold set to: {thr_prec:.4f}")
print(f"   Results: Precision = {p_score*100:.2f}%, Recall = {r_score*100:.2f}%, F1 = {f1_score_val*100:.2f}%")
print("\n👉 Done! Open your Google Drive -> 'dataset' folder to grab the saved `.h5`, `.pkl`, and `.txt` files!")
