"""
03_preprocessing_no_leakage.py
Prepares train/test splits for XGBoost training (no data leakage).
Uses scale_pos_weight instead of undersampling to preserve all legitimate data.
Saves numpy arrays to preprocessed_data_no_leakage/
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_PATH = '../engineered_data.csv'
OUT_DIR    = '../preprocessed_data_no_leakage'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature set: 24 legitimate features (no leakage, no manual heuristics) ──
FEATURE_COLS = [
    'amount', 'logAmount', 'amountToMeanRatio', 'isHighAmount',
    'amountBin', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT',
    'type_TRANSFER', 'typeEncoded', 'isHighRiskType', 'isTransfer', 'isCashOut',
    'origIsCustomer', 'destIsMerchant', 'destIsCustomer', 'isC2C', 'isC2M',
    'hourOfDay', 'dayNumber', 'isNightTime', 'dayOfWeek', 'isWeekend'
]
# Dropped: step (simulator artefact), highRiskCombo, suspiciousTransfer,
#          muleIndicator, fraudRiskScore (manual heuristics that leak signal)
TARGET = 'isFraud'

print("Loading engineered data...")
df = pd.read_csv(INPUT_PATH)
print(f"Shape: {df.shape}")

X = df[FEATURE_COLS].values
y = df[TARGET].values

fraud_count = y.sum()
legit_count = (y == 0).sum()
ratio = legit_count / fraud_count
print(f"Fraud: {fraud_count:,}  Legit: {legit_count:,}  Ratio: {ratio:.1f}:1")
print(f"Recommended scale_pos_weight: {ratio:.0f}")

# ── Train/test split (stratified) ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

# ── Scale features ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Save arrays ──────────────────────────────────────────────────────────────
np.save(f'{OUT_DIR}/X_train.npy',        X_train)
np.save(f'{OUT_DIR}/X_test.npy',         X_test)
np.save(f'{OUT_DIR}/X_train_scaled.npy', X_train_scaled)
np.save(f'{OUT_DIR}/X_test_scaled.npy',  X_test_scaled)
np.save(f'{OUT_DIR}/y_train.npy',        y_train)
np.save(f'{OUT_DIR}/y_test.npy',         y_test)

with open(f'{OUT_DIR}/feature_columns_24.pkl', 'wb') as f:
    pickle.dump(FEATURE_COLS, f)

class_weights = {0: 1.0, 1: ratio}
with open(f'{OUT_DIR}/class_weights.pkl', 'wb') as f:
    pickle.dump(class_weights, f)

print(f"\n✅ Saved preprocessing outputs to {OUT_DIR}/")
print(f"   scale_pos_weight to use in XGBoost: {ratio:.0f}")
