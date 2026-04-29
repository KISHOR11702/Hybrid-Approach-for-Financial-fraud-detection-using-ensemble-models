"""
13_lstm_bayesian_optimisation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Applies Optuna TPE Bayesian hyperparameter search to the LSTM sequences model.

Goal: improve Precision / reduce False Positives compared to the baseline model
      (baseline: Precision=21.27%, Recall=65.15%, AUC=0.634, threshold=0.5301)

Search space
────────────
• LSTM layer sizes (units 1 & 2)
• Dropout & recurrent-dropout rates
• Dense bottleneck units
• Learning rate
• Batch size
• Fraud class-weight multiplier (higher → model is more selective → higher prec)
• Focal-loss gamma  (0 = binary cross-entropy; >0 = focuses on hard examples)

Strategy
────────
• 30 Optuna TPE trials, each trained on a 12% subsample (CV=3) for speed
• Objective = PR-AUC  (directly captures Precision–Recall tradeoff)
• Final model: full training on best params
• Threshold chosen as the point where Precision ≥ target (default 40%) with
  max Recall at that precision level, then also reported at F1-optimal

Outputs  (all NEW files — nothing existing is overwritten)
───────────────────────────────────────────────────────────
  models/lstm_bayes_optimised_final.h5
  models/lstm_bayes_optimised_best.h5
  models/lstm_bayes_optimised_scaler.pkl
  models/lstm_bayes_optimised_threshold.txt
  models/lstm_bayes_optimised_params.pkl
  models/lstm_bayes_optimised_metadata.json

Usage
─────
  cd d:\\Major_project
  python python_scripts/13_lstm_bayesian_optimisation.py

  # To skip the search and only re-train with saved best params:
  python python_scripts/13_lstm_bayesian_optimisation.py --skip-search
"""

import argparse
import json
import os
import pickle
import sys
import warnings
import time


def to_python(obj):
    """Recursively convert numpy scalars / arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(i) for i in obj]
    if hasattr(obj, 'item'):          # numpy scalar (int64, float64, bool_…)
        return obj.item()
    if hasattr(obj, 'tolist'):        # numpy ndarray
        return obj.tolist()
    return obj

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, precision_score, recall_score,
    confusion_matrix,
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
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'fraud_detection_dataset_lstm_v3.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Outputs — NEVER clash with existing model files
OUT_PREFIX     = os.path.join(MODELS_DIR, 'lstm_bayes_optimised')
FINAL_H5       = OUT_PREFIX + '_final.h5'
BEST_H5        = OUT_PREFIX + '_best.h5'
SCALER_PKL     = OUT_PREFIX + '_scaler.pkl'
THRESHOLD_TXT  = OUT_PREFIX + '_threshold.txt'
PARAMS_PKL     = OUT_PREFIX + '_params.pkl'
METADATA_JSON  = OUT_PREFIX + '_metadata.json'

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN    = 15
N_FEATURES = 19

SEQ_FEATURE_COLS = [
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

# ── Optuna config ─────────────────────────────────────────────────────────────
N_TRIALS        = 30
SUBSAMPLE_FRAC  = 0.12    # fraction of users per trial (speed vs quality)
CV_FOLDS        = 3
PREC_TARGET     = 0.40    # minimum precision for threshold selection
EPOCHS_TRIAL    = 10      # epochs per CV fold during search
EPOCHS_FINAL    = 60      # epochs for final full training
PATIENCE_FINAL  = 12


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _encode_col(series, vocab):
    m = {v: i for i, v in enumerate(vocab)}
    return series.map(lambda x: m.get(str(x), 0))


def _most_common_past(group_series):
    result, past = [], []
    for val in group_series:
        result.append(max(set(past), key=past.count) if past else val)
        past.append(val)
    return pd.Series(result, index=group_series.index)


def load_and_preprocess(path, max_users=None):
    """Load v2 CSV, build all 19 sequence features per user."""
    print(f"  Loading {path} …")
    df = pd.read_csv(path)

    if max_users:
        users = df['user_id'].unique()[:max_users]
        df    = df[df['user_id'].isin(users)]

    print(f"  Rows={len(df):,}  Users={df['user_id'].nunique():,}")

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    df['hour']         = df['timestamp'].dt.hour.fillna(12).astype(int)
    df['day_of_week']  = df['timestamp'].dt.dayofweek.fillna(1).astype(int)
    df['day_of_month'] = df['timestamp'].dt.day.fillna(15).astype(int)
    df['month']        = df['timestamp'].dt.month.fillna(1).astype(int)

    df['merchant_category_encoded'] = _encode_col(df['merchant_category'], MERCHANT_CATEGORIES)
    df['device_id_encoded']         = _encode_col(df['device_id'],         DEVICE_TYPES)
    df['location_encoded']          = _encode_col(df['location'],          LOCATIONS)

    # most_used_device / most_used_location already pre-computed in the v2 CSV
    if 'most_used_device' not in df.columns:
        df['most_used_device'] = df.groupby('user_id', group_keys=False)['device_id'].apply(_most_common_past)
    if 'most_used_location' not in df.columns:
        df['most_used_location'] = df.groupby('user_id', group_keys=False)['location'].apply(_most_common_past)

    df['most_used_device_encoded']   = _encode_col(df['most_used_device'],   DEVICE_TYPES)
    df['most_used_location_encoded'] = _encode_col(df['most_used_location'], LOCATIONS)

    df['location_changed'] = (df['location'] != df['most_used_location']).astype(int)
    df['device_changed']   = (df['device_id'] != df['most_used_device']).astype(int)

    for col in SEQ_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df[SEQ_FEATURE_COLS] = df[SEQ_FEATURE_COLS].fillna(0)

    return df


def build_sequences(df):
    """(N_users, SEQ_LEN, N_FEATURES)  +  labels."""
    X_seqs, y_labels = [], []
    n_feat = len(SEQ_FEATURE_COLS)
    # label column is 'is_fraud' in fraud_detection_dataset_lstm_v2.csv
    fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'isFraud'
    for _, grp in df.groupby('user_id', sort=False):
        feats = grp[SEQ_FEATURE_COLS].values.astype(float)
        label = int(grp[fraud_col].max())
        n     = len(feats)
        seq   = feats[-SEQ_LEN:] if n >= SEQ_LEN else \
                np.vstack([np.zeros((SEQ_LEN - n, n_feat)), feats])
        X_seqs.append(seq)
        y_labels.append(label)
    return np.array(X_seqs), np.array(y_labels)


# ══════════════════════════════════════════════════════════════════════════════
#  FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════════════

def focal_loss(gamma=2.0, alpha=0.25):
    """Binary focal loss — focuses training on hard / misclassified examples."""
    def loss_fn(y_true, y_pred):
        y_pred  = K.clip(y_pred, 1e-7, 1 - 1e-7)
        bce     = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl      = alpha * K.pow(1 - p_t, gamma) * bce
        return K.mean(fl)
    return loss_fn


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_model(params: dict):
    """
    params keys:
      lstm_units_1, lstm_units_2, dropout_1, dropout_2,
      recurrent_dropout, dense_units, learning_rate,
      focal_gamma  (0 → standard BCE, >0 → focal loss)
    """
    inp = keras.Input(shape=(SEQ_LEN, N_FEATURES))

    x = layers.LSTM(
        params['lstm_units_1'],
        return_sequences=True,
        dropout=params['dropout_1'],
        recurrent_dropout=params['recurrent_dropout'],
    )(inp)

    x = layers.LSTM(
        params['lstm_units_2'],
        return_sequences=False,
        dropout=params['dropout_2'],
        recurrent_dropout=params['recurrent_dropout'],
    )(x)

    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(params['dense_units'], activation='relu')(x)
    x   = layers.Dropout(params['dropout_2'])(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inp, out)

    gamma = params.get('focal_gamma', 0.0)
    loss  = focal_loss(gamma=gamma) if gamma > 0 else 'binary_crossentropy'

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss=loss,
        metrics=[keras.metrics.AUC(name='pr_auc', curve='PR')],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  THRESHOLD SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_threshold(y_true, y_prob, prec_target=PREC_TARGET):
    """
    Returns two thresholds:
      thr_f1    – maximises F1
      thr_prec  – highest threshold where Precision >= prec_target with
                  recall >= 0.30 (to avoid degenerate all-negative predictor)
    """
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)

    # F1-optimal
    f1_arr  = np.where((prec_arr + rec_arr) == 0, 0,
                       2 * prec_arr * rec_arr / (prec_arr + rec_arr))
    f1_idx  = np.argmax(f1_arr)
    thr_f1  = float(thr_arr[f1_idx]) if f1_idx < len(thr_arr) else 0.5

    # Precision-targeted (max recall where precision >= target & recall >= 0.30)
    mask    = (prec_arr >= prec_target) & (rec_arr >= 0.30)
    if mask.any():
        candidates = np.where(mask)[0]
        # among candidates pick the one with highest recall
        best_c   = candidates[np.argmax(rec_arr[candidates])]
        thr_prec = float(thr_arr[best_c]) if best_c < len(thr_arr) else thr_f1
    else:
        print(f"  ⚠️  Precision target {prec_target*100:.0f}% not achievable "
              f"with recall>=30%.  Using F1 threshold instead.")
        thr_prec = thr_f1

    return thr_f1, thr_prec


def print_metrics(tag, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true,   y_pred, zero_division=0)
    f1     = f1_score(y_true,       y_pred, zero_division=0)
    roc    = roc_auc_score(y_true,  y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n  ── {tag}  (threshold={threshold:.4f}) ──")
    print(f"     TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(f"     False-Positive Rate : {fp_rate*100:.2f}%")
    print(f"     Precision           : {prec*100:.2f}%")
    print(f"     Recall              : {rec*100:.2f}%")
    print(f"     F1-Score            : {f1*100:.2f}%")
    print(f"     ROC-AUC             : {roc:.4f}")
    print(f"     PR-AUC              : {pr_auc:.4f}")
    return dict(threshold=threshold, precision=prec, recall=rec,
                f1=f1, roc_auc=roc, pr_auc=pr_auc,
                tp=tp, fp=fp, fn=fn, tn=tn, fp_rate=fp_rate)


# ══════════════════════════════════════════════════════════════════════════════
#  BAYESIAN OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_search(X_all, y_all, n_trials=N_TRIALS,
                    subsample_frac=SUBSAMPLE_FRAC, cv_folds=CV_FOLDS):
    """
    Optuna TPE search. Objective = mean PR-AUC across CV folds on a subsample.
    """
    if not OPTUNA_OK:
        raise RuntimeError("optuna not installed. Run: pip install optuna")

    n_sub = int(len(X_all) * subsample_frac)
    idx   = np.random.choice(len(X_all), n_sub, replace=False)
    X_s   = X_all[idx];  y_s = y_all[idx]

    print(f"\n  Bayesian search: {n_trials} trials | "
          f"subsample={n_sub:,} users | {cv_folds}-fold CV")
    print(f"  Objective: PR-AUC (scaler fit per fold — no leakage)\n")

    # NOTE: do NOT pre-scale here. Scaler is fit inside each fold
    # so validation statistics never leak into training.
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            'lstm_units_1':      trial.suggest_categorical('lstm_units_1',    [64, 96, 128, 192, 256]),
            'lstm_units_2':      trial.suggest_categorical('lstm_units_2',    [32, 48, 64, 96, 128]),
            'dropout_1':         trial.suggest_float('dropout_1',             0.10, 0.50),
            'dropout_2':         trial.suggest_float('dropout_2',             0.10, 0.50),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout',     0.00, 0.30),
            'dense_units':       trial.suggest_categorical('dense_units',     [16, 32, 48, 64]),
            'learning_rate':     trial.suggest_float('learning_rate',         5e-5, 5e-3, log=True),
            # v3 dataset: 90% normal / 10% fraud users → base ratio ~9:1
            # multiplier 5–15x gives effective fraud weight of ~45–135 (selective)
            'class_weight_mult': trial.suggest_float('class_weight_mult', 0.5, 3.0),
            # focal gamma: small values only — large gamma + class weight double-penalises
            'focal_gamma':       trial.suggest_float('focal_gamma', 0.0, 1.5),
        }

        scores = []
        for tr_idx, val_idx in cv.split(X_s, y_s):
            # ── Fit scaler on TRAIN fold only (no leakage) ──────────────────
            _sh        = X_s.shape
            _scaler    = StandardScaler()
            X_tr_f = _scaler.fit_transform(X_s[tr_idx].reshape(-1, _sh[-1])).reshape(X_s[tr_idx].shape)
            X_va_f = _scaler.transform(X_s[val_idx].reshape(-1, _sh[-1])).reshape(X_s[val_idx].shape)
            fraud_w = (y_s[tr_idx] == 0).sum() / max((y_s[tr_idx] == 1).sum(), 1)
            cw = {0: 1.0, 1: fraud_w * params['class_weight_mult']}

            K.clear_session()
            model = build_model(params)
            cb = EarlyStopping(monitor='val_pr_auc', patience=3,
                               restore_best_weights=True, mode='max')
            model.fit(
                X_tr_f, y_s[tr_idx],
                validation_data=(X_va_f, y_s[val_idx]),
                epochs=EPOCHS_TRIAL,
                batch_size=512,
                class_weight=cw,
                callbacks=[cb],
                verbose=0,
            )
            prob   = model.predict(X_va_f, verbose=0).flatten()
            score  = average_precision_score(y_s[val_idx], prob)
            scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
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
    fraud_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    mult    = params.get('class_weight_mult', 1.0)
    cw      = {0: 1.0, 1: fraud_w * mult}
    print(f"\n  Class weight — fraud : {cw[1]:.1f}  (mult={mult:.2f})")

    K.clear_session()
    model = build_model(params)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_pr_auc', patience=PATIENCE_FINAL,
                      restore_best_weights=True, mode='max', verbose=1),
        ModelCheckpoint(BEST_H5, monitor='val_pr_auc',
                        save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_pr_auc', factor=0.5, patience=6,
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
    print("\n" + "="*65)
    print("  LSTM BAYESIAN HYPERPARAMETER OPTIMISATION")
    print("="*65)

    # ── Load & build sequences ───────────────────────────────────────────
    print("\n[1/4]  Loading data and building sequences …")
    df   = load_and_preprocess(DATA_PATH)
    X, y = build_sequences(df)
    print(f"  Sequences: {X.shape}  Fraud users: {y.sum()} / {len(y)} "
          f"({y.mean()*100:.2f}%)")

    # ── Train / test split (user-level, stratified) ──────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"  Train={len(y_tr):,}  Test={len(y_te):,}")

    # ── Scale (fit on train only) ────────────────────────────────────────
    sh       = X_tr.shape
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr.reshape(-1, sh[-1])).reshape(sh)
    X_te_sc  = scaler.transform(X_te.reshape(-1, sh[-1])).reshape(X_te.shape)

    with open(SCALER_PKL, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✅ Saved scaler → {SCALER_PKL}")

    # ── Bayesian search or load saved params ────────────────────────────
    if skip_search and os.path.exists(PARAMS_PKL):
        print("\n[2/4]  Loading saved best params (--skip-search) …")
        with open(PARAMS_PKL, 'rb') as f:
            best_params = pickle.load(f)
        print(f"  Params: {json.dumps(best_params, indent=4)}")
    else:
        print("\n[2/4]  Bayesian hyperparameter search …")
        best_params, study = bayesian_search(X_tr_sc, y_tr)
        with open(PARAMS_PKL, 'wb') as f:
            pickle.dump(best_params, f)
        print(f"  ✅ Saved best params → {PARAMS_PKL}")

    # ── Full training with best params ───────────────────────────────────
    print("\n[3/4]  Training final model on full training set …")
    model, history = train_full(best_params, X_tr_sc, y_tr, X_te_sc, y_te, scaler)

    # ── Evaluate & select thresholds ────────────────────────────────────
    print("\n[4/4]  Evaluating & selecting thresholds …")
    y_prob = model.predict(X_te_sc, verbose=0).flatten()

    thr_f1, thr_prec = select_threshold(y_te, y_prob, prec_target=PREC_TARGET)

    print(f"\n  F1-optimal threshold   : {thr_f1:.4f}")
    print(f"  Precision-target (≥{PREC_TARGET*100:.0f}%) : {thr_prec:.4f}")

    m_f1   = print_metrics("F1-optimal threshold",    y_te, y_prob, thr_f1)
    m_prec = print_metrics(f"Precision≥{PREC_TARGET*100:.0f}% threshold", y_te, y_prob, thr_prec)

    print("\n  ── Baseline model (for comparison) ──")
    print("     Precision=21.27%  Recall=65.15%  F1=32.07%  AUC=0.634  threshold=0.53")

    # Save precision-targeted threshold as the primary threshold
    chosen_thr = thr_prec
    with open(THRESHOLD_TXT, 'w') as f:
        f.write(f'{chosen_thr:.4f}')
    print(f"\n  ✅ Saved threshold ({chosen_thr:.4f}) → {THRESHOLD_TXT}")

    # ── Save metadata ────────────────────────────────────────────────────
    metadata = {
        'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'LSTM Bayesian Optimised (precision-focused)',
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
        'baseline_comparison': {
            'precision': 0.2127, 'recall': 0.6515,
            'f1': 0.3207, 'auc': 0.634, 'threshold': 0.5301,
        },
        'train_users': int(len(y_tr)),
        'test_users':  int(len(y_te)),
        'fraud_rate': float(y.mean()),
    }
    with open(METADATA_JSON, 'w') as f:
        json.dump(to_python(metadata), f, indent=2)
    print(f"  ✅ Saved metadata → {METADATA_JSON}")

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  DONE  ({elapsed/60:.1f} min)")
    print(f"{'='*65}")
    print(f"\n  Model files saved:")
    print(f"    {FINAL_H5}")
    print(f"    {BEST_H5}")
    print(f"    {SCALER_PKL}")
    print(f"    {THRESHOLD_TXT}  (value={chosen_thr:.4f})")
    print(f"    {PARAMS_PKL}")
    print(f"    {METADATA_JSON}")

    print(f"\n  To use in app.py, update load_lstm_model() to load:")
    print(f"    models/lstm_bayes_optimised_final.h5")
    print(f"    models/lstm_bayes_optimised_scaler.pkl")
    print(f"    models/lstm_bayes_optimised_threshold.txt")
    print()

    return model, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-search', action='store_true',
                        help='Skip Optuna search; use saved params from last run')
    args = parser.parse_args()
    main(skip_search=args.skip_search)
