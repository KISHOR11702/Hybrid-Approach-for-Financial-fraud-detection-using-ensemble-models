"""
12_generate_test_dataset_10k.py
Generates a 10,000-row PaySim-format test dataset with realistic fraud signals.
Samples from the full PaySim CSV and ensures fraud transactions are present.
Output: uploads/test_dataset_10k.csv
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

SOURCE_PATH = '../transaction_data.csv'
OUTPUT_PATH = '../uploads/test_dataset_10k.csv'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

TARGET_ROWS  = 10_000
FRAUD_TARGET = 500    # 5% fraud rate for evaluation convenience

print(f"Loading {SOURCE_PATH} ...")
df = pd.read_csv(SOURCE_PATH)
print(f"Source shape: {df.shape}")

fraud_df = df[df['isFraud'] == 1]
legit_df = df[df['isFraud'] == 0]

print(f"Fraud available: {len(fraud_df):,}  Legit: {len(legit_df):,}")

# Sample
n_fraud = min(FRAUD_TARGET, len(fraud_df))
n_legit = TARGET_ROWS - n_fraud

fraud_sample = fraud_df.sample(n=n_fraud, random_state=42)
legit_sample = legit_df.sample(n=n_legit, random_state=42)

out = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=42)
out.reset_index(drop=True, inplace=True)

out.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved {len(out):,} rows → {OUTPUT_PATH}")
print(f"   Fraud: {out['isFraud'].sum()} ({out['isFraud'].mean()*100:.2f}%)")
print(f"   Type distribution:\n{out['type'].value_counts().to_string()}")
