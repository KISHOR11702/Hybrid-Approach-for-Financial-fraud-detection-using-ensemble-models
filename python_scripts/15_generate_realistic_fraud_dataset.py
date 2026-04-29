"""
15_generate_realistic_fraud_dataset.py
Generates a realistic LSTM test dataset with subtle fraud signals.
Designed so fraud is detectable but not trivially obvious (no 8x spikes).
  - 600 users, 20-35 transactions each
  - ~18% fraud users; last 2-4 tx are fraudulent per fraud user
  - Amount spike: 1.5x-3x running average (not 2.5x-8x)
  - Location change probability for fraud: 65%
  - Device change probability for fraud:   60%
  - Running past-mean avg_transaction_amount (no future leakage)
Output: uploads/lstm_test_dataset.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

OUTPUT_PATH  = '../uploads/lstm_test_dataset.csv'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

N_USERS          = 600
TX_MIN, TX_MAX   = 20, 35
FRAUD_USER_RATE  = 0.18

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

rows = []
base_time = datetime(2024, 6, 1, 9, 0, 0)

for uid in range(N_USERS):
    user_id  = f'USER_{uid:04d}'
    n_tx     = random.randint(TX_MIN, TX_MAX)
    is_fraud_user = uid < int(N_USERS * FRAUD_USER_RATE)

    home_loc    = random.choice(LOCATIONS)
    home_device = random.choice(DEVICE_TYPES)
    base_amount = random.uniform(1_000, 20_000)
    acct_age    = random.randint(90, 1800)

    fraud_start  = n_tx - random.randint(2, 4)
    current_time = base_time + timedelta(days=random.randint(0, 180),
                                          hours=random.randint(0, 23))
    past_amounts = []   # running accumulator

    for tx_i in range(n_tx):
        is_fraud = is_fraud_user and tx_i >= fraud_start

        # ── Timing ──────────────────────────────────────────────────
        gap_hrs = random.expovariate(1.0 / 8)
        current_time += timedelta(hours=gap_hrs)

        # ── Running past mean (no current row included) ─────────────
        past_avg = float(np.mean(past_amounts)) if past_amounts else base_amount

        # ── Amount ──────────────────────────────────────────────────
        if is_fraud:
            amount = past_avg * random.uniform(1.5, 3.0)
        else:
            amount = max(50, np.random.lognormal(np.log(base_amount), 0.5))
        amount = round(amount, 2)

        dev_pct = (amount - past_avg) / (past_avg + 1e-9) * 100
        past_amounts.append(amount)

        # ── Location / device ────────────────────────────────────────
        if is_fraud:
            alt_locs = [l for l in LOCATIONS if l != home_loc]
            location = random.choice(alt_locs) if (random.random() < 0.65 and alt_locs) else home_loc
            alt_devs = [d for d in DEVICE_TYPES if d != home_device]
            device   = random.choice(alt_devs) if (random.random() < 0.60 and alt_devs) else home_device
        else:
            location = home_loc   if random.random() < 0.88 else random.choice(LOCATIONS)
            device   = home_device if random.random() < 0.92 else random.choice(DEVICE_TYPES)

        merchant = 'ATM Withdrawal' if is_fraud and random.random() < 0.35 \
            else random.choice(MERCHANT_CATEGORIES)

        # ── Tx count features ─────────────────────────────────────────
        # Simplified: count past rows within time windows
        tx_1hr  = sum(1 for r in rows
                      if r['user_id'] == user_id
                      and (current_time - pd.to_datetime(r['timestamp'])).total_seconds() <= 3600)
        tx_24hr = sum(1 for r in rows
                      if r['user_id'] == user_id
                      and (current_time - pd.to_datetime(r['timestamp'])).total_seconds() <= 86400)

        rows.append({
            'user_id':                      user_id,
            'transaction_id':               f'TX_{uid:04d}_{tx_i:03d}',
            'timestamp':                    current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'amount':                       amount,
            'merchant_category':            merchant,
            'location':                     location,
            'device_id':                    device,
            'account_age_days':             acct_age + tx_i,
            'avg_transaction_amount':       round(past_avg, 2),
            'amount_deviation_from_avg_pct': round(dev_pct, 2),
            'transaction_frequency':        tx_i + 1,
            'time_gap_from_previous_hrs':   round(gap_hrs, 3),
            'transactions_last_1hr':        tx_1hr,
            'transactions_last_24hr':       tx_24hr,
            'isFraud':                      int(is_fraud),
        })

df = pd.DataFrame(rows)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour']        = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_month'] = df['timestamp'].dt.day
df['month']       = df['timestamp'].dt.month
df['timestamp']   = df['timestamp'].astype(str)

fraud_users = df.groupby('user_id')['isFraud'].max().sum()
print(f"Rows           : {len(df):,}")
print(f"Users          : {df['user_id'].nunique()}")
print(f"Fraud users    : {fraud_users}")
print(f"Fraud txns     : {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")
print(f"Avg amount dev — normal: {df[df['isFraud']==0]['amount_deviation_from_avg_pct'].mean():.2f}%")
print(f"Avg amount dev — fraud : {df[df['isFraud']==1]['amount_deviation_from_avg_pct'].mean():.2f}%")

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved → {OUTPUT_PATH}")
