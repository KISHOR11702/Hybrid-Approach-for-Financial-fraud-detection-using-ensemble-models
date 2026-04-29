"""
14_generate_advanced_fraud_dataset.py
Generates a large synthetic v2-format dataset with rich behavioural features
for training the LSTM sequences model.
Includes: user_id, location, device_id, merchant_category, timestamps,
          running averages, time-gap features, fraud signals.
Output: fraud_detection_dataset_lstm_v2.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

OUTPUT_PATH = '../fraud_detection_dataset_lstm_v2.csv'
N_USERS     = 5_000
TX_PER_USER_MIN = 20
TX_PER_USER_MAX = 60
FRAUD_USER_RATE = 0.15    # 15% of users commit fraud

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
base_time = datetime(2024, 1, 1, 8, 0, 0)

print(f"Generating dataset for {N_USERS:,} users ...")

for uid in range(N_USERS):
    user_id  = f'USER_{uid:06d}'
    n_tx     = random.randint(TX_PER_USER_MIN, TX_PER_USER_MAX)
    is_fraud_user = uid < int(N_USERS * FRAUD_USER_RATE)

    home_loc    = random.choice(LOCATIONS)
    home_device = random.choice(DEVICE_TYPES)
    avg_amount  = random.uniform(500, 25_000)
    account_age = random.randint(30, 2000)

    fraud_start = n_tx - random.randint(2, 4)   # last 2-4 txns are fraud
    current_time = base_time + timedelta(days=random.randint(0, 365))
    running_amounts = []

    for tx_i in range(n_tx):
        is_fraud = is_fraud_user and tx_i >= fraud_start

        # Timing
        gap_hrs = random.expovariate(1.0 / 12)   # avg 12-hr gap
        current_time += timedelta(hours=gap_hrs)

        # Amount
        if is_fraud:
            amount = avg_amount * random.uniform(1.5, 3.0)
        else:
            amount = max(10, np.random.lognormal(np.log(avg_amount), 0.6))

        # Deviation from running avg
        past_avg = np.mean(running_amounts) if running_amounts else amount
        dev_pct  = (amount - past_avg) / (past_avg + 1e-9) * 100
        running_amounts.append(amount)

        # Location / device
        if is_fraud:
            location  = random.choice([l for l in LOCATIONS  if l != home_loc])   if random.random() < 0.65 else home_loc
            device_id = random.choice([d for d in DEVICE_TYPES if d != home_device]) if random.random() < 0.60 else home_device
        else:
            location  = home_loc   if random.random() < 0.85 else random.choice(LOCATIONS)
            device_id = home_device if random.random() < 0.90 else random.choice(DEVICE_TYPES)

        merchant_category = 'ATM Withdrawal' if is_fraud and random.random() < 0.3 \
            else random.choice(MERCHANT_CATEGORIES)

        rows.append({
            'user_id':                     user_id,
            'transaction_id':              f'TX_{uid:06d}_{tx_i:04d}',
            'timestamp':                   current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'amount':                      round(amount, 2),
            'merchant_category':           merchant_category,
            'location':                    location,
            'device_id':                   device_id,
            'account_age_days':            account_age + tx_i,
            'avg_transaction_amount':      round(past_avg, 2),
            'amount_deviation_from_avg_pct': round(dev_pct, 2),
            'transaction_frequency':       tx_i + 1,
            'time_gap_from_previous_hrs':  round(gap_hrs, 3),
            'transactions_last_1hr':       0,   # simplified
            'transactions_last_24hr':      0,
            'isFraud':                     int(is_fraud),
        })

df = pd.DataFrame(rows)
# compute hour features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour']       = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_month'] = df['timestamp'].dt.day
df['month']       = df['timestamp'].dt.month
df['timestamp']   = df['timestamp'].astype(str)

fraud_users = df.groupby('user_id')['isFraud'].max().sum()
print(f"Total rows  : {len(df):,}")
print(f"Fraud users : {fraud_users:,} / {N_USERS:,} ({fraud_users/N_USERS*100:.1f}%)")
print(f"Fraud txns  : {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved → {OUTPUT_PATH}")
