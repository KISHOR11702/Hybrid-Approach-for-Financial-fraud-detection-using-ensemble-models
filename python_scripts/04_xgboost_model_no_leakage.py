"""
04_xgboost_model_no_leakage.py
Trains XGBoost fraud detector on PaySim data.
  - 24 leak-free features (step + 4 manual heuristics removed)
  - scale_pos_weight=774 (true class imbalance, no undersampling)
  - Bayesian hyperparameter optimisation via Optuna (TPE, PR-AUC objective)
  - Saves model, threshold, and metrics to models/

Training output (Bayesian-optimised, threshold=0.60):
    Precision = 5.72%  |  Recall = 78.49%  |  F1 = 10.66%
    ROC-AUC   = 0.9745 |  PR-AUC = 0.5054
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              precision_recall_curve, f1_score,
                              precision_score, recall_score)
import xgboost as xgb

# ── Optuna (install: pip install optuna) ──────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  optuna not installed. Run: pip install optuna")
    print("   Bayesian optimisation will be skipped.")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = '../engineered_data.csv'   # output of 02_feature_engineering.py
MODELS_DIR = '../models'
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Features ──────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'amount', 'logAmount', 'amountToMeanRatio', 'isHighAmount',
    'amountBin', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT',
    'type_TRANSFER', 'typeEncoded', 'isHighRiskType', 'isTransfer', 'isCashOut',
    'origIsCustomer', 'destIsMerchant', 'destIsCustomer', 'isC2C', 'isC2M',
    'hourOfDay', 'dayNumber', 'isNightTime', 'dayOfWeek', 'isWeekend'
]
# Explicitly removed (would cause leakage or inflate performance):
REMOVE_FEATURES = ['step', 'highRiskCombo', 'suspiciousTransfer',
                   'muleIndicator', 'fraudRiskScore']

TARGET = 'isFraud'

# ── Default hyperparameters (used when Bayesian search is skipped) ─────────────
DEFAULT_PARAMS = {
    'n_estimators':   300,
    'max_depth':      4,
    'learning_rate':  0.05,
    'subsample':      0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma':          0.1,
    'reg_alpha':      0.1,
    'reg_lambda':     1.0,
    'scale_pos_weight': 774,   # legit:fraud ≈ 773:1 in PaySim full dataset
    'eval_metric':    'aucpr',
    'use_label_encoder': False,
    'random_state':   42,
    'n_jobs':         -1,
}


def load_data():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")

    # Keep only the 24 legitimate features
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠️  Missing features (will be zeroed): {missing}")
        for c in missing:
            df[c] = 0

    X = df[available].values
    y = df[TARGET].values
    fraud = y.sum()
    legit = (y == 0).sum()
    print(f"  Fraud: {fraud:,}  Legit: {legit:,}  Ratio: {legit/fraud:.1f}:1")
    return X, y, available


def find_best_threshold(y_true, y_prob):
    """Return threshold that maximises F1 on the given labels/probs."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s  = np.where((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
    best = np.argmax(f1s)
    return float(thr[best]) if best < len(thr) else 0.5


def evaluate(model, X_test, y_test, threshold):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    prec    = precision_score(y_test, y_pred,    zero_division=0)
    rec     = recall_score(y_test, y_pred,        zero_division=0)
    f1      = f1_score(y_test, y_pred,            zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)

    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tp = int(((y_pred == 1) & (y_test == 1)).sum())

    print(f"\n  Threshold : {threshold:.4f}")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return {'precision': prec, 'recall': rec, 'f1': f1,
            'roc_auc': roc_auc, 'pr_auc': pr_auc, 'threshold': threshold}


def bayesian_optimize(X_train, y_train, n_trials=40,
                      subsample_frac=0.15, cv_folds=3):
    """Optuna TPE search; objective = mean PR-AUC across CV folds."""
    if not OPTUNA_AVAILABLE:
        print("Skipping Bayesian optimisation (optuna not installed).")
        return DEFAULT_PARAMS

    print(f"\nRunning Bayesian optimisation: {n_trials} trials, "
          f"{cv_folds}-fold CV, subsample={subsample_frac*100:.0f}% ...")

    # subsample to speed up each trial
    n_sub = int(len(X_train) * subsample_frac)
    idx   = np.random.choice(len(X_train), n_sub, replace=False)
    X_s, y_s = X_train[idx], y_train[idx]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'max_depth':        trial.suggest_int('max_depth', 3, 6),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma':            trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 5.0),
            'scale_pos_weight': 774,
            'eval_metric':      'aucpr',
            'use_label_encoder': False,
            'random_state':     42,
            'n_jobs':           -1,
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_s, y_s):
            model = xgb.XGBClassifier(**params)
            model.fit(X_s[tr_idx], y_s[tr_idx],
                      eval_set=[(X_s[val_idx], y_s[val_idx])],
                      verbose=False)
            prob   = model.predict_proba(X_s[val_idx])[:, 1]
            scores.append(average_precision_score(y_s[val_idx], prob))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({'scale_pos_weight': 774, 'eval_metric': 'aucpr',
                 'use_label_encoder': False, 'random_state': 42, 'n_jobs': -1})
    print(f"\n  Best PR-AUC (CV): {study.best_value:.4f}")
    print(f"  Best params: {best}")
    return best


def run_training_pipeline(run_bayesian=True, n_trials=40):
    X, y, feat_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

    # ── Hyperparameter optimisation ──────────────────────────────────────
    if run_bayesian:
        best_params = bayesian_optimize(X_train, y_train,
                                        n_trials=n_trials,
                                        subsample_frac=0.15,
                                        cv_folds=3)
    else:
        best_params = DEFAULT_PARAMS
        print(f"\nUsing default params: {best_params}")

    # ── Full training ────────────────────────────────────────────────────
    print("\nTraining final model on full training set ...")
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=50)

    # ── Threshold ────────────────────────────────────────────────────────
    y_prob_train = model.predict_proba(X_train)[:, 1]
    threshold    = find_best_threshold(y_train, y_prob_train)
    threshold    = max(threshold, 0.60)   # floor at 0.60 to control FPs
    print(f"\nOptimal threshold: {threshold:.4f}")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n=== Test-set evaluation ===")
    metrics = evaluate(model, X_test, y_test, threshold)

    # ── Feature importance ───────────────────────────────────────────────
    imp = pd.Series(model.feature_importances_, index=feat_names)
    print("\n=== Top-10 Feature Importances ===")
    print(imp.sort_values(ascending=False).head(10).to_string())

    # ── Save ─────────────────────────────────────────────────────────────
    model_path  = f'{MODELS_DIR}/xgboost_model_no_leakage.pkl'
    thresh_path = f'{MODELS_DIR}/xgboost_model_no_leakage_threshold.pkl'
    met_path    = f'{MODELS_DIR}/xgboost_model_no_leakage_metrics.pkl'

    with open(model_path,  'wb') as f: pickle.dump(model, f)
    with open(thresh_path, 'wb') as f: pickle.dump({'threshold': threshold}, f)
    with open(met_path,    'wb') as f: pickle.dump(metrics, f)

    print(f"\n✅ Saved: {model_path}")
    print(f"✅ Saved: {thresh_path}")
    print(f"✅ Saved: {met_path}")

    # ── Text report ──────────────────────────────────────────────────────
    report = (
        "XGBoost Fraud Detection — No-Leakage Model\n"
        "=" * 50 + "\n"
        f"Features      : {len(feat_names)}\n"
        f"Threshold     : {threshold:.4f}\n"
        f"Precision     : {metrics['precision']*100:.2f}%\n"
        f"Recall        : {metrics['recall']*100:.2f}%\n"
        f"F1-Score      : {metrics['f1']*100:.2f}%\n"
        f"ROC-AUC       : {metrics['roc_auc']:.4f}\n"
        f"PR-AUC        : {metrics['pr_auc']:.4f}\n"
    )
    print("\n" + report)
    with open('../xgboost_report_no_leakage.txt', 'w') as f:
        f.write(report)

    return model, metrics


if __name__ == '__main__':
    run_training_pipeline(run_bayesian=True, n_trials=40)
