"""
02_feature_engineering.py
Engineers all fraud-detection features from the raw PaySim dataset.
Saves to engineered_data.csv for use by downstream training scripts.
"""

import pandas as pd
import numpy as np
import os

DATA_PATH   = '../transaction_data.csv'
OUTPUT_PATH = '../engineered_data.csv'

print("Loading raw data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

# ── Remove leakage columns (balance diffs reveal fraud directly) ──────────────
LEAKAGE_COLS = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                'newbalanceDest', 'isFlaggedFraud']
df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], inplace=True)

# ── Amount features ───────────────────────────────────────────────────────────
df['logAmount']          = np.log1p(df['amount'])
df['amountToMeanRatio']  = df['amount'] / (df['amount'].mean() + 1e-10)
df['isHighAmount']       = (df['amount'] > df['amount'].quantile(0.75)).astype(int)
df['amountBin']          = pd.cut(df['amount'], bins=10, labels=False,
                                   duplicates='drop').fillna(0).astype(int)

# ── Transaction type one-hot & label encoding ─────────────────────────────────
type_dummies = pd.get_dummies(df['type'], prefix='type')
for col in ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']:
    if col not in type_dummies.columns:
        type_dummies[col] = 0
df = pd.concat([df, type_dummies[['type_CASH_IN','type_CASH_OUT','type_DEBIT',
                                   'type_PAYMENT','type_TRANSFER']]], axis=1)

type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
df['typeEncoded']    = df['type'].map(type_map).fillna(0).astype(int)
df['isHighRiskType'] = df['type'].isin(['CASH_OUT', 'TRANSFER']).astype(int)
df['isTransfer']     = (df['type'] == 'TRANSFER').astype(int)
df['isCashOut']      = (df['type'] == 'CASH_OUT').astype(int)

# ── Account-type flags ────────────────────────────────────────────────────────
df['origIsCustomer'] = df['nameOrig'].str.startswith('C').astype(int)
df['destIsMerchant'] = df['nameDest'].str.startswith('M').astype(int)
df['destIsCustomer'] = df['nameDest'].str.startswith('C').astype(int)
df['isC2C']          = (df['origIsCustomer'] & df['destIsCustomer']).astype(int)
df['isC2M']          = (df['origIsCustomer'] & df['destIsMerchant']).astype(int)

# ── Temporal features (step = hours since simulation start) ──────────────────
df['hourOfDay']   = df['step'] % 24
df['dayNumber']   = df['step'] // 24
df['isNightTime'] = ((df['hourOfDay'] >= 23) | (df['hourOfDay'] < 5)).astype(int)
df['dayOfWeek']   = (df['step'] // 24) % 7
df['isWeekend']   = df['dayOfWeek'].isin([5, 6]).astype(int)

# ── Manual heuristics (kept for reference; dropped before XGBoost training) ──
df['highRiskCombo']      = (df['isHighAmount'] & df['isHighRiskType'] & df['isNightTime']).astype(int)
df['suspiciousTransfer'] = (df['isTransfer']   & df['isHighAmount']).astype(int)
df['muleIndicator']      = (df['isC2C']        & df['isHighAmount']).astype(int)
df['fraudRiskScore']     = (df['isHighRiskType'] * 3
                            + df['isHighAmount']  * 2
                            + df['isNightTime']   * 1)

print(f"\nEngineered features: {df.shape[1]} columns")
print(df.dtypes.to_string())

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved engineered data → {OUTPUT_PATH}")
