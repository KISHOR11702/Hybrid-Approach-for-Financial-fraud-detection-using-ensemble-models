"""
01_data_loading_eda.py
Exploratory Data Analysis on the PaySim fraud detection dataset.
Loads the raw CSV, inspects distributions, fraud rates, and saves plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = '../transaction_data.csv'   # adjust if needed
PLOTS_DIR  = '../plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.dtypes)
print(df.head(3))

# ── Basic stats ──────────────────────────────────────────────────────────────
print("\n=== Missing values ===")
print(df.isnull().sum())

print("\n=== isFraud distribution ===")
print(df['isFraud'].value_counts())
fraud_rate = df['isFraud'].mean() * 100
print(f"Fraud rate: {fraud_rate:.4f}%")

print("\n=== Transaction types ===")
print(df['type'].value_counts())

# ── Plot 1: Fraud distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
df['isFraud'].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'crimson'])
ax.set_title('Fraud vs Legitimate Transactions')
ax.set_xlabel('isFraud (0=Legit, 1=Fraud)')
ax.set_ylabel('Count')
ax.set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/01_fraud_distribution.png', dpi=150)
plt.close()
print("Saved: 01_fraud_distribution.png")

# ── Plot 2: Transaction types ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
type_fraud = df.groupby('type')['isFraud'].mean() * 100
type_fraud.sort_values(ascending=False).plot(kind='bar', ax=ax, color='coral')
ax.set_title('Fraud Rate by Transaction Type (%)')
ax.set_xlabel('Transaction Type')
ax.set_ylabel('Fraud Rate (%)')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/02_transaction_types.png', dpi=150)
plt.close()
print("Saved: 02_transaction_types.png")

# ── Plot 3: Amount distribution ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[df['isFraud'] == 0]['amount'].clip(upper=1e6).hist(bins=50, ax=axes[0], color='steelblue', alpha=0.7)
axes[0].set_title('Amount Distribution — Legitimate')
df[df['isFraud'] == 1]['amount'].clip(upper=1e6).hist(bins=50, ax=axes[1], color='crimson', alpha=0.7)
axes[1].set_title('Amount Distribution — Fraud')
for ax in axes:
    ax.set_xlabel('Amount')
    ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/03_amount_distribution.png', dpi=150)
plt.close()
print("Saved: 03_amount_distribution.png")

# ── Plot 4: Correlation heatmap ───────────────────────────────────────────────
numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest', 'isFraud']
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/04_correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: 04_correlation_heatmap.png")

# ── Plot 5: Fraud over time ───────────────────────────────────────────────────
fraud_by_step = df[df['isFraud'] == 1].groupby('step').size()
fig, ax = plt.subplots(figsize=(14, 4))
fraud_by_step.plot(ax=ax, color='crimson', linewidth=0.8)
ax.set_title('Fraud Transactions Over Time (by step)')
ax.set_xlabel('Step (hours)')
ax.set_ylabel('Fraud Count')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/05_fraud_over_time.png', dpi=150)
plt.close()
print("Saved: 05_fraud_over_time.png")

print("\n✅ EDA complete. All plots saved to:", PLOTS_DIR)
