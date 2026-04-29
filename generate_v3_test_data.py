import pandas as pd
import os

print("Loading v3 Dataset...")
v3_path = 'd:/Major_project/fraud_detection_dataset_lstm_v3.csv'
if not os.path.exists(v3_path):
    print(f"Error: Could not find dataset at {v3_path}")
else:
    df = pd.read_csv(v3_path)
    
    # Sort chronologically by user first!
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values(['user_id', 'timestamp'])
    
    # We want a continuous chunk of users to preserve LSTM sequences (15 per user)
    unique_users = df['user_id'].unique()
    
    # Grab the first 2,000 users (roughly 15,000 transactions)
    test_users = unique_users[:2000]
    test_df = df[df['user_id'].isin(test_users)].copy()
    
    print(f"Sampled {len(test_df)} transactions.")
    fraud_count = test_df['is_fraud'].sum()
    print(f"Frauds in this sample: {fraud_count}")
    
    # Save to uploads folder
    out_path = 'd:/Major_project/uploads/v3_hybrid_test_file.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    test_df.to_csv(out_path, index=False)
    print(f"✅ Saved perfectly sequenced test file to: {out_path}")
    print("🚀 Go upload this file into the Web App!")
