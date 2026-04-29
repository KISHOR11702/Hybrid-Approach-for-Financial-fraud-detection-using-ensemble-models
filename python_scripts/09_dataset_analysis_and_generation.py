"""
09_dataset_analysis_and_generation.py
Analyses the raw PaySim and v2 datasets side-by-side and reports key statistics.
Also verifies the generated LSTM test dataset for correct fraud signal ratios.
"""

import pandas as pd
import numpy as np
import os

PAYSIM_PATH  = '../transaction_data.csv'
V2_PATH      = '../fraud_detection_dataset_lstm_v2.csv'
TEST_PATH    = '../uploads/test_dataset_10k.csv'


def analyse(path, label):
    if not os.path.exists(path):
        print(f"[{label}] File not found: {path}")
        return
    df = pd.read_csv(path, nrows=200_000)   # cap for speed
    print(f"\n{'='*55}")
    print(f"  {label}: {path}")
    print(f"{'='*55}")
    print(f"  Shape       : {df.shape}")
    print(f"  Columns     : {list(df.columns)}")
    if 'isFraud' in df.columns:
        fr = df['isFraud'].mean() * 100
        print(f"  Fraud rate  : {fr:.4f}%  ({df['isFraud'].sum()} frauds)")
    if 'type' in df.columns:
        print(f"  Type dist   :")
        print(df['type'].value_counts().to_string(indent=4))
    if 'amount' in df.columns:
        print(f"  Amount  min={df['amount'].min():.2f}  "
              f"max={df['amount'].max():.2f}  "
              f"mean={df['amount'].mean():.2f}")


if __name__ == '__main__':
    analyse(PAYSIM_PATH,  'PaySim Dataset')
    analyse(V2_PATH,      'Fraud Detection v2 (LSTM)')
    analyse(TEST_PATH,    'Test Dataset 10k')

    # ── Validate LSTM test dataset ─────────────────────────────────────────
    v2_test = '../uploads/lstm_test_dataset.csv'
    if os.path.exists(v2_test):
        df = pd.read_csv(v2_test)
        fraud_users = df[df['isFraud'] == 1]['user_id'].nunique() \
            if 'user_id' in df.columns else 0
        total_users = df['user_id'].nunique() if 'user_id' in df.columns else 0
        print(f"\n=== LSTM Test Dataset ===")
        print(f"  Rows         : {len(df)}")
        print(f"  Total users  : {total_users}")
        print(f"  Fraud users  : {fraud_users} ({fraud_users/max(total_users,1)*100:.1f}%)")
        print(f"  Fraud txns   : {df['isFraud'].sum()} "
              f"({df['isFraud'].mean()*100:.2f}%)")

        if 'amount' in df.columns and 'isFraud' in df.columns:
            print(f"  Mean amount — fraud : {df[df['isFraud']==1]['amount'].mean():.2f}")
            print(f"  Mean amount — legit : {df[df['isFraud']==0]['amount'].mean():.2f}")

    print("\n✅ Dataset analysis complete.")
