"""
16_generate_balanced_dataset.py
────────────────────────────────
Generates a balanced synthetic v3 dataset for LSTM training.

Distribution (user-level):
    Normal users  : 90%  (108,000)
    Fraud  users  : 10%  (12,000)

Column schema matches fraud_detection_dataset_lstm_v2.csv exactly:
    transaction_id, user_id, timestamp, amount, merchant_category,
    device_id, location, time_gap_from_previous_hrs,
    amount_deviation_from_avg_pct, transactions_last_1hr,
    transactions_last_24hr, account_age_days, avg_transaction_amount,
    transaction_frequency, most_used_device, most_used_location, is_fraud

Output:  fraud_detection_dataset_lstm_v3.csv  (~3.6 M rows)
Usage:
    cd d:\\Major_project
    python python_scripts/16_generate_balanced_dataset.py
"""

import os
import random
import time
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH      = os.path.join(BASE_DIR, 'fraud_detection_dataset_lstm_v3.csv')

N_USERS          = 120_000
NORMAL_RATE      = 0.90   # 90 % normal users
FRAUD_RATE       = 0.10   # 10 % fraud  users

TX_MIN_NORMAL    = 20     # txns per normal user
TX_MAX_NORMAL    = 50
TX_MIN_FRAUD     = 25     # fraud users have slightly more history
TX_MAX_FRAUD     = 55

FRAUD_TX_COUNT   = (2, 5) # how many of the user's txns are fraud (trailing)

CHUNK_WRITE      = 50_000  # flush rows to CSV every N users (memory control)

# ── Vocab ─────────────────────────────────────────────────────────────────────
MERCHANT_CATEGORIES = [
    'Grocery', 'Restaurant', 'Gas Station', 'Online Shopping', 'Electronics',
    'Pharmacy', 'Entertainment', 'Travel', 'Utilities', 'Insurance',
    'Healthcare', 'Education', 'Clothing', 'Home Improvement', 'ATM Withdrawal'
]
DEVICE_TYPES = [
    'Mobile_iOS', 'Mobile_Android', 'Desktop_Windows',
    'Desktop_Mac', 'Tablet_iOS', 'Tablet_Android', 'Web_Browser'
]
LOCATIONS = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata',
    'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Chandigarh',
    'Bhopal', 'Nagpur', 'Vadodara', 'Coimbatore', 'Indore', 'Patna'
]

BASE_TIME = datetime(2024, 1, 1, 8, 0, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  PER-USER GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_user(uid_int, is_fraud_user: bool) -> list[dict]:
    user_id     = f'USER_{uid_int:06d}'
    n_tx        = random.randint(TX_MIN_FRAUD, TX_MAX_FRAUD) if is_fraud_user \
                  else random.randint(TX_MIN_NORMAL, TX_MAX_NORMAL)
    home_loc    = random.choice(LOCATIONS)
    home_device = random.choice(DEVICE_TYPES)
    avg_amount  = random.uniform(300, 30_000)
    account_age = random.randint(30, 2_500)

    # How many trailing transactions are fraud
    n_fraud_tx  = random.randint(*FRAUD_TX_COUNT) if is_fraud_user else 0
    fraud_start = n_tx - n_fraud_tx

    current_time   = BASE_TIME + timedelta(days=random.randint(0, 365),
                                           hours=random.randint(0, 23))
    running_amounts: list[float] = []
    # Track device/location history for most_used computation
    loc_history:    list[str] = []
    dev_history:    list[str] = []
    # sliding-window counters (simple step-based)
    timestamps:     list[datetime] = []

    rows = []
    tx_global_id = uid_int * 100   # unique-ish across users

    for tx_i in range(n_tx):
        is_fraud_tx = is_fraud_user and (tx_i >= fraud_start)

        # ── Time gap ──────────────────────────────────────────────────────
        if is_fraud_tx:
            # Realistic fraud: slightly faster than normal but NOT burst-like.
            # Use exponential with avg 6 hrs — overlaps heavily with normal
            # (normal avg=14 hrs, so short gaps occur in both classes)
            gap_hrs = random.expovariate(1.0 / 6)
            gap_hrs = max(0.25, min(gap_hrs, 48.0))   # cap extremes
        else:
            gap_hrs = random.expovariate(1.0 / 14)   # avg 14-hr gap
        current_time += timedelta(hours=gap_hrs)
        timestamps.append(current_time)

        # ── Amount ────────────────────────────────────────────────────────
        if is_fraud_tx:
            # Only 60 % of fraud txns have elevated amounts;
            # 40 % look completely normal (mimics real-world card fraud)
            if random.random() < 0.60:
                mult   = random.uniform(1.10, 2.50)   # subtle: 10-150% above avg
            else:
                mult   = random.uniform(0.60, 1.10)   # indistinguishable from normal
            amount = avg_amount * mult
        else:
            amount = max(10.0, np.random.lognormal(np.log(avg_amount), 0.55))

        past_avg = float(np.mean(running_amounts)) if running_amounts else amount
        dev_pct  = (amount - past_avg) / (past_avg + 1e-9) * 100
        running_amounts.append(amount)

        # ── Location & Device ─────────────────────────────────────────────
        if is_fraud_tx:
            # Realistic: only mildly higher chance of unusual location/device
            # (many fraud txns happen on the victim's own device/location)
            location  = random.choice([l for l in LOCATIONS   if l != home_loc])   \
                        if random.random() < 0.35 else home_loc
            device_id = random.choice([d for d in DEVICE_TYPES if d != home_device]) \
                        if random.random() < 0.30 else home_device
        else:
            location  = home_loc    if random.random() < 0.82 else random.choice(LOCATIONS)
            device_id = home_device if random.random() < 0.88 else random.choice(DEVICE_TYPES)

        # ── Merchant ──────────────────────────────────────────────────────
        # Subtle bias only — fraud spread across all categories, slight skew
        if is_fraud_tx and random.random() < 0.20:
            merchant_category = 'ATM Withdrawal'
        elif is_fraud_tx and random.random() < 0.20:
            merchant_category = random.choice(['Electronics', 'Online Shopping', 'Travel'])
        else:
            merchant_category = random.choice(MERCHANT_CATEGORIES)

        # ── most_used device / location (historical mode up to current tx) ─
        most_used_dev = Counter(dev_history).most_common(1)[0][0] \
                        if dev_history else device_id
        most_used_loc = Counter(loc_history).most_common(1)[0][0] \
                        if loc_history else location
        loc_history.append(location)
        dev_history.append(device_id)

        # ── Sliding-window counts ─────────────────────────────────────────
        one_hr_ago  = current_time - timedelta(hours=1)
        one_day_ago = current_time - timedelta(hours=24)
        tx_last_1hr  = sum(1 for t in timestamps[:-1] if t >= one_hr_ago)
        tx_last_24hr = sum(1 for t in timestamps[:-1] if t >= one_day_ago)

        rows.append({
            'transaction_id':               f'TXN_{tx_global_id + tx_i:09d}',
            'user_id':                      user_id,
            'timestamp':                    current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'amount':                       round(amount, 2),
            'merchant_category':            merchant_category,
            'device_id':                    device_id,
            'location':                     location,
            'time_gap_from_previous_hrs':   round(gap_hrs, 4),
            'amount_deviation_from_avg_pct': round(dev_pct, 4),
            'transactions_last_1hr':        tx_last_1hr,
            'transactions_last_24hr':       tx_last_24hr,
            'account_age_days':             account_age + tx_i,
            'avg_transaction_amount':       round(past_avg, 2),
            'transaction_frequency':        tx_i + 1,
            'most_used_device':             most_used_dev,
            'most_used_location':           most_used_loc,
            'is_fraud':                     int(is_fraud_tx),
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("  BALANCED DATASET GENERATOR  (v3)")
    print("=" * 60)

    n_normal = int(N_USERS * NORMAL_RATE)   # 108,000
    n_fraud  = N_USERS - n_normal           # 12,000

    # Shuffle user indices so fraud users are distributed, not all at the end
    user_indices = list(range(N_USERS))
    random.shuffle(user_indices)
    fraud_set = set(user_indices[:n_fraud])

    print(f"\n  Total users  : {N_USERS:,}")
    print(f"  Normal users : {n_normal:,}  ({NORMAL_RATE*100:.0f}%)")
    print(f"  Fraud  users : {n_fraud:,}  ({FRAUD_RATE*100:.0f}%)")
    print(f"  Output       : {OUTPUT_PATH}")
    print()

    header_written = False
    total_rows     = 0
    total_fraud_tx = 0
    buffer         = []

    for idx, uid_int in enumerate(range(N_USERS)):
        is_fraud_user = uid_int in fraud_set
        rows = generate_user(uid_int, is_fraud_user)
        buffer.extend(rows)

        # Progress
        if (idx + 1) % 10_000 == 0:
            elapsed = time.time() - t0
            pct = (idx + 1) / N_USERS * 100
            print(f"  [{pct:5.1f}%]  Users={idx+1:,}  "
                  f"Rows≈{total_rows + len(buffer):,}  "
                  f"Elapsed={elapsed:.0f}s")

        # Flush to disk every CHUNK_WRITE users
        if len(buffer) >= CHUNK_WRITE * 30:  # ~30 txns avg per user
            df_chunk = pd.DataFrame(buffer)
            total_rows     += len(df_chunk)
            total_fraud_tx += int(df_chunk['is_fraud'].sum())
            df_chunk.to_csv(OUTPUT_PATH,
                            mode='w' if not header_written else 'a',
                            header=not header_written,
                            index=False)
            header_written = True
            buffer = []

    # Final flush
    if buffer:
        df_chunk = pd.DataFrame(buffer)
        total_rows     += len(df_chunk)
        total_fraud_tx += int(df_chunk['is_fraud'].sum())
        df_chunk.to_csv(OUTPUT_PATH,
                        mode='w' if not header_written else 'a',
                        header=not header_written,
                        index=False)

    elapsed = time.time() - t0
    fraud_tx_pct = total_fraud_tx / total_rows * 100 if total_rows else 0

    print(f"\n{'='*60}")
    print(f"  DONE  ({elapsed:.1f}s)")
    print(f"  Total rows         : {total_rows:,}")
    print(f"  Normal txns        : {total_rows - total_fraud_tx:,}  ({100 - fraud_tx_pct:.2f}%)")
    print(f"  Fraud  txns        : {total_fraud_tx:,}  ({fraud_tx_pct:.2f}%)")
    print(f"  Normal users       : {n_normal:,}  (90.00%)")
    print(f"  Fraud  users       : {n_fraud:,}  (10.00%)")
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"  File size          : {size_mb:.1f} MB")
    print(f"  Saved → {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
