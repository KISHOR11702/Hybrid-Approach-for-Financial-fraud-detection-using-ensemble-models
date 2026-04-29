"""
fraud_dataset_leakage_analysis.py
Analyses the PaySim dataset for data leakage sources.
Identifies features that directly or indirectly reveal the fraud label,
which would inflate model performance during evaluation.
Reports feature importance with and without leakage features.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

DATA_PATH  = '../engineered_data.csv'
MODELS_DIR = '../models'

# ── Known leakage features ────────────────────────────────────────────────────
LEAKAGE_FEATURES = [
    'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'errorBalanceOrig', 'errorBalanceDest', 'hasBalanceError',
    'amountToBalanceRatioOrig', 'amountToBalanceRatioDest',
    'balanceChangePercOrig', 'origBalanceZero', 'destBalanceZero',
    'origNewBalanceZero', 'isFlaggedFraud',
]

# ── Suspect manual heuristics ─────────────────────────────────────────────────
HEURISTIC_FEATURES = [
    'step',              # simulation counter — not a real-world feature
    'highRiskCombo',     # derived from isFraud-correlated features
    'suspiciousTransfer',
    'muleIndicator',
    'fraudRiskScore',
]

# ── 24 legitimate features ────────────────────────────────────────────────────
LEGIT_FEATURES = [
    'amount', 'logAmount', 'amountToMeanRatio', 'isHighAmount',
    'amountBin', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT',
    'type_TRANSFER', 'typeEncoded', 'isHighRiskType', 'isTransfer', 'isCashOut',
    'origIsCustomer', 'destIsMerchant', 'destIsCustomer', 'isC2C', 'isC2M',
    'hourOfDay', 'dayNumber', 'isNightTime', 'dayOfWeek', 'isWeekend'
]
TARGET = 'isFraud'


def quick_xgb_auc(X_tr, X_te, y_tr, y_te, label):
    m = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        use_label_encoder=False, eval_metric='aucpr',
        random_state=42, n_jobs=-1
    )
    m.fit(X_tr, y_tr, verbose=False)
    prob = m.predict_proba(X_te)[:, 1]
    auc  = roc_auc_score(y_te, prob)
    print(f"  [{label}]  ROC-AUC = {auc:.4f}  (features={X_tr.shape[1]})")
    return m, auc


if __name__ == '__main__':
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")

    y = df[TARGET].values

    print("\n" + "="*60)
    print("LEAKAGE ANALYSIS: Comparing ROC-AUC with/without leak features")
    print("="*60)

    # 1. With leakage features
    leak_cols = [c for c in LEAKAGE_FEATURES if c in df.columns]
    if leak_cols:
        X_leak = df[leak_cols].fillna(0).values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_leak, y, test_size=0.3, random_state=42, stratify=y)
        quick_xgb_auc(X_tr, X_te, y_tr, y_te, 'Leakage features only')
    else:
        print("  No leakage columns found in dataset (already removed).")

    # 2. With heuristic features (step + manual)
    heur_cols = [c for c in HEURISTIC_FEATURES + LEGIT_FEATURES if c in df.columns]
    X_heur = df[heur_cols].fillna(0).values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_heur, y, test_size=0.3, random_state=42, stratify=y)
    m_heur, _ = quick_xgb_auc(X_tr, X_te, y_tr, y_te,
                               'Legit + heuristics (step, etc.)')

    # Show step's importance
    heur_imp = pd.Series(m_heur.feature_importances_, index=heur_cols)
    if 'step' in heur_imp.index:
        step_pct = heur_imp['step'] / heur_imp.sum() * 100
        print(f"\n  ⚠️  'step' feature importance: {step_pct:.1f}% "
              f"— this is a simulation artefact, NOT a real-world signal!")

    # 3. Legitimate features only (no leakage, no heuristics)
    legit_cols = [c for c in LEGIT_FEATURES if c in df.columns]
    X_legit = df[legit_cols].fillna(0).values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_legit, y, test_size=0.3, random_state=42, stratify=y)
    m_legit, _ = quick_xgb_auc(X_tr, X_te, y_tr, y_te,
                                '24 legitimate features only ✅')

    legit_imp = pd.Series(m_legit.feature_importances_, index=legit_cols)
    print("\n=== Top-10 Legitimate Feature Importances ===")
    print(legit_imp.sort_values(ascending=False).head(10).to_string())

    print("\n=== Conclusion ===")
    print("  Use ONLY the 24 legitimate features for production training.")
    print("  Leakage and heuristic features inflate metrics during evaluation.")
    print(f"  Features to REMOVE: {REMOVE_R := HEURISTIC_FEATURES}")
    print(f"  Features to KEEP  : {legit_cols}")
