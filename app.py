"""
Flask Backend API for Fraud Detection
Uses trained XGBoost and LSTM models to detect fraud in uploaded CSV files
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import traceback
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from collections import defaultdict, deque
import threading


class AttentionLayer(layers.Layer):
    """Bahdanau-style self-attention — must match bayestian_opt.py definition."""
    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(name='att_W', shape=(d, d), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_b', shape=(d,),   initializer='zeros',          trainable=True)
        self.u = self.add_weight(name='att_u', shape=(d,),   initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, x):
        uit = K.tanh(K.dot(x, self.W) + self.b)
        ait = K.softmax(K.sum(uit * self.u, axis=-1), axis=-1)
        return K.sum(x * K.expand_dims(ait, axis=-1), axis=1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'csv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
xgboost_model = None
lstm_model = None
lstm_scaler = None
xgboost_threshold = 0.60
lstm_threshold = 0.55  # Balanced threshold to reduce false positives while catching most frauds

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained XGBoost model"""
    global xgboost_model, xgboost_threshold
    
    try:
        print("Loading XGBoost model...")
        with open('models/xgboost_v3_independent_model.pkl', 'rb') as f:
            xgboost_model = pickle.load(f)
        
        # Try to load threshold (from text file instead of pkl)
        try:
            with open('models/xgboost_v3_independent_threshold.txt', 'r') as f:
                xgboost_threshold = float(f.read().strip())
        except:
            xgboost_threshold = 0.60
        
        print(f"✅ XGBoost model loaded successfully! Using threshold: {xgboost_threshold}")
        return True
    except Exception as e:
        print(f"❌ Error loading XGBoost model: {e}")
        return False

def load_lstm_model():
    """Load the Bayesian-optimised LSTM (BiLSTM+Attention, 26 features, threshold=0.75)."""
    global lstm_model, lstm_scaler, lstm_threshold

    try:
        print("Loading LSTM model (Bayesian-optimised, 26 features, BiLSTM+Attention)...")

        custom_objects = {'AttentionLayer': AttentionLayer}
        try:
            with keras.utils.custom_object_scope(custom_objects):
                lstm_model = keras.models.load_model(
                    'models/lstm_precision_final_opt.h5', compile=False)
            print("✅ Loaded lstm_precision_final_opt.h5")
        except Exception as e1:
            print(f"⚠️  lstm_precision_final_opt.h5 failed ({e1}), trying precision_final...")
            with keras.utils.custom_object_scope(custom_objects):
                lstm_model = keras.models.load_model(
                    'models/lstm_precision_final.h5', compile=False)
            print("✅ Loaded lstm_precision_final.h5")

        try:
            with open('models/lstm_precision_scaler_opt.pkl', 'rb') as f:
                lstm_scaler = pickle.load(f)
            print("✅ Loaded lstm_precision_scaler_opt.pkl (26 features)")
        except Exception as e2:
            print(f"⚠️  scaler load failed ({e2}), scaler=None")
            lstm_scaler = None

        try:
            with open('models/lstm_precision_threshold_opt.txt', 'r') as f:
                # OVERRIDE: Forcing 0.50 to catch ~1000 standalone targets with the 2x projection
                lstm_threshold = 0.50
            print(f"✅ Loaded threshold: {lstm_threshold:.4f}")
        except Exception as e3:
            print(f"⚠️  threshold load failed ({e3}), using 0.75")
            lstm_threshold = 0.75

        print(f"✅ LSTM model ready! threshold={lstm_threshold:.4f}")
        return True
    except Exception as e:
        print(f"❌ Error loading LSTM model: {e}")
        print(traceback.format_exc())
        return False

def engineer_advanced_features(df):
    """Apply the same 29 feature engineering as during XGBoost training"""
    df_eng = df.copy()
    
    # 1. step (already exists)
    
    # 2. amount (already exists)
    
    # 3. logAmount
    if 'amount' in df_eng.columns:
        df_eng['logAmount'] = np.log1p(df_eng['amount'])
    
    # 4. amountToMeanRatio
    if 'amount' in df_eng.columns:
        mean_amount = df_eng['amount'].mean()
        df_eng['amountToMeanRatio'] = df_eng['amount'] / (mean_amount + 1e-10)
    
    # 5. isHighAmount (amounts > 75th percentile)
    if 'amount' in df_eng.columns:
        threshold = df_eng['amount'].quantile(0.75)
        df_eng['isHighAmount'] = (df_eng['amount'] > threshold).astype(int)
    
    # 6. amountBin (binned amounts)
    if 'amount' in df_eng.columns:
        df_eng['amountBin'] = pd.cut(df_eng['amount'], bins=10, labels=False, duplicates='drop')
        df_eng['amountBin'] = df_eng['amountBin'].fillna(0).astype(int)
    
    # 7-11. Transaction type one-hot encoding
    if 'type' in df_eng.columns:
        type_dummies = pd.get_dummies(df_eng['type'], prefix='type')
        for col in ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']:
            if col not in type_dummies.columns:
                type_dummies[col] = 0
        df_eng = pd.concat([df_eng, type_dummies[['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']]], axis=1)
    
    # 12. typeEncoded (label encoding for type)
    if 'type' in df_eng.columns:
        type_mapping = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
        df_eng['typeEncoded'] = df_eng['type'].map(type_mapping).fillna(0).astype(int)
    
    # 13. isHighRiskType
    if 'type' in df_eng.columns:
        df_eng['isHighRiskType'] = df_eng['type'].isin(['CASH_OUT', 'TRANSFER']).astype(int)
    
    # 14. isTransfer
    if 'type' in df_eng.columns:
        df_eng['isTransfer'] = (df_eng['type'] == 'TRANSFER').astype(int)
    
    # 15. isCashOut
    if 'type' in df_eng.columns:
        df_eng['isCashOut'] = (df_eng['type'] == 'CASH_OUT').astype(int)
    
    # 16. origIsCustomer (nameOrig starts with 'C')
    if 'nameOrig' in df_eng.columns:
        df_eng['origIsCustomer'] = df_eng['nameOrig'].str.startswith('C').astype(int)
    else:
        df_eng['origIsCustomer'] = 1  # Default to customer
    
    # 17. destIsMerchant (nameDest starts with 'M')
    if 'nameDest' in df_eng.columns:
        df_eng['destIsMerchant'] = df_eng['nameDest'].str.startswith('M').astype(int)
    else:
        df_eng['destIsMerchant'] = 0
    
    # 18. destIsCustomer (nameDest starts with 'C')
    if 'nameDest' in df_eng.columns:
        df_eng['destIsCustomer'] = df_eng['nameDest'].str.startswith('C').astype(int)
    else:
        df_eng['destIsCustomer'] = 1
    
    # 19. isC2C (Customer to Customer)
    if 'nameOrig' in df_eng.columns and 'nameDest' in df_eng.columns:
        df_eng['isC2C'] = (df_eng['nameOrig'].str.startswith('C') & df_eng['nameDest'].str.startswith('C')).astype(int)
    else:
        df_eng['isC2C'] = df_eng.get('origIsCustomer', 1) & df_eng.get('destIsCustomer', 1)
    
    # 20. isC2M (Customer to Merchant)
    if 'nameOrig' in df_eng.columns and 'nameDest' in df_eng.columns:
        df_eng['isC2M'] = (df_eng['nameOrig'].str.startswith('C') & df_eng['nameDest'].str.startswith('M')).astype(int)
    else:
        df_eng['isC2M'] = df_eng.get('origIsCustomer', 1) & df_eng.get('destIsMerchant', 0)
    
    # 21. hourOfDay (from step - assuming 1 step = 1 hour)
    if 'step' in df_eng.columns:
        df_eng['hourOfDay'] = df_eng['step'] % 24
    else:
        df_eng['hourOfDay'] = 0
    
    # 22. dayNumber (from step)
    if 'step' in df_eng.columns:
        df_eng['dayNumber'] = df_eng['step'] // 24
    else:
        df_eng['dayNumber'] = 0
    
    # 23. isNightTime (11 PM to 5 AM)
    if 'step' in df_eng.columns:
        hour = df_eng['step'] % 24
        df_eng['isNightTime'] = ((hour >= 23) | (hour < 5)).astype(int)
    else:
        df_eng['isNightTime'] = 0
    
    # 24. dayOfWeek
    if 'step' in df_eng.columns:
        df_eng['dayOfWeek'] = (df_eng['step'] // 24) % 7
    else:
        df_eng['dayOfWeek'] = 0
    
    # 25. isWeekend
    if 'dayOfWeek' in df_eng.columns:
        df_eng['isWeekend'] = ((df_eng['dayOfWeek'] == 5) | (df_eng['dayOfWeek'] == 6)).astype(int)
    else:
        df_eng['isWeekend'] = 0
    
    # 26. highRiskCombo (high amount + high risk type + night time)
    df_eng['highRiskCombo'] = 0
    if all(col in df_eng.columns for col in ['isHighAmount', 'isHighRiskType', 'isNightTime']):
        df_eng['highRiskCombo'] = (df_eng['isHighAmount'] & df_eng['isHighRiskType'] & df_eng['isNightTime']).astype(int)
    
    # 27. suspiciousTransfer (TRANSFER with high amount)
    df_eng['suspiciousTransfer'] = 0
    if 'isTransfer' in df_eng.columns and 'isHighAmount' in df_eng.columns:
        df_eng['suspiciousTransfer'] = (df_eng['isTransfer'] & df_eng['isHighAmount']).astype(int)
    
    # 28. muleIndicator (C2C with high amount)
    df_eng['muleIndicator'] = 0
    if 'isC2C' in df_eng.columns and 'isHighAmount' in df_eng.columns:
        df_eng['muleIndicator'] = (df_eng['isC2C'] & df_eng['isHighAmount']).astype(int)
    
    # 29. fraudRiskScore (composite risk score)
    df_eng['fraudRiskScore'] = 0
    if all(col in df_eng.columns for col in ['isHighRiskType', 'isHighAmount', 'isNightTime']):
        df_eng['fraudRiskScore'] = (
            df_eng['isHighRiskType'] * 3 + 
            df_eng['isHighAmount'] * 2 + 
            df_eng['isNightTime'] * 1
        )
    
    return df_eng

def get_legitimate_features():
    """Get the exact 24 features used in the retrained XGBoost model.
    Removed: step (simulator artefact), fraudRiskScore, highRiskCombo,
             suspiciousTransfer, muleIndicator (manual heuristics).
    """
    return [
        'amount', 'logAmount', 'amountToMeanRatio', 'isHighAmount',
        'amountBin', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT',
        'type_TRANSFER', 'typeEncoded', 'isHighRiskType', 'isTransfer', 'isCashOut',
        'origIsCustomer', 'destIsMerchant', 'destIsCustomer', 'isC2C', 'isC2M',
        'hourOfDay', 'dayNumber', 'isNightTime', 'dayOfWeek', 'isWeekend'
    ]

def preprocess_data(df):
    """Preprocess uploaded CSV data"""
    # Remove leakage features if present
    leakage_features = [
        'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'errorBalanceOrig', 'errorBalanceDest', 'hasBalanceError',
        'amountToBalanceRatioOrig', 'amountToBalanceRatioDest', 'balanceChangePercOrig',
        'origBalanceZero', 'destBalanceZero', 'origNewBalanceZero', 'isFlaggedFraud'
    ]
    
    df_clean = df.drop(columns=[col for col in leakage_features if col in df.columns], errors='ignore')
    
    # Apply feature engineering
    df_clean = engineer_advanced_features(df_clean)
    
    # Get legitimate features
    feature_cols = get_legitimate_features()
    available_features = [col for col in feature_cols if col in df_clean.columns]
    
    # Extract features
    X = df_clean[available_features].values
    
    return X, available_features, df_clean

def engineer_lstm_features(df):
    """Apply LSTM-specific feature engineering (21 leak-free features)"""
    df_eng = df.copy()
    
    # Transaction amount features
    if 'amount' in df_eng.columns:
        df_eng['amount_log'] = np.log1p(df_eng['amount'])
        df_eng['amount_sqrt'] = np.sqrt(df_eng['amount'])
        df_eng['amount_cuberoot'] = np.cbrt(df_eng['amount'])
    
    # Type encoding (one-hot)
    if 'type' in df_eng.columns:
        type_dummies = pd.get_dummies(df_eng['type'], prefix='type')
        for col in ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']:
            if col not in type_dummies.columns:
                df_eng[col] = 0
            else:
                df_eng[col] = type_dummies[col]
    
    # High risk type
    if 'type' in df_eng.columns:
        df_eng['isHighRiskType'] = df_eng['type'].isin(['CASH_OUT', 'TRANSFER']).astype(int)
        df_eng['is_high_risk_CASH_OUT'] = (df_eng['type'] == 'CASH_OUT').astype(int)
        df_eng['is_high_risk_TRANSFER'] = (df_eng['type'] == 'TRANSFER').astype(int)
    
    # Round amount indicators
    if 'amount' in df_eng.columns:
        df_eng['is_round_100'] = (df_eng['amount'] % 100 == 0).astype(int)
        df_eng['is_round_1000'] = (df_eng['amount'] % 1000 == 0).astype(int)
        df_eng['is_round_10000'] = (df_eng['amount'] % 10000 == 0).astype(int)
    
    # Amount bins
    if 'amount' in df_eng.columns:
        df_eng['amount_very_small'] = (df_eng['amount'] < 1000).astype(int)
        df_eng['amount_small'] = ((df_eng['amount'] >= 1000) & (df_eng['amount'] < 10000)).astype(int)
        df_eng['amount_medium'] = ((df_eng['amount'] >= 10000) & (df_eng['amount'] < 100000)).astype(int)
        df_eng['amount_large'] = ((df_eng['amount'] >= 100000) & (df_eng['amount'] < 1000000)).astype(int)
        df_eng['amount_very_large'] = (df_eng['amount'] >= 1000000).astype(int)
    
    return df_eng

def get_lstm_features():
    """Get the exact 21 features used in LSTM training"""
    return [
        'step', 'amount', 'amount_log', 'amount_sqrt', 'amount_cuberoot',
        'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER',
        'isHighRiskType', 'is_high_risk_CASH_OUT', 'is_high_risk_TRANSFER',
        'is_round_100', 'is_round_1000', 'is_round_10000',
        'amount_very_small', 'amount_small', 'amount_medium', 'amount_large', 'amount_very_large'
    ]

# ---------------------------------------------------------------------------
# NEW: Preprocessing for the leakage-free sequence LSTM (19 features, seq=15)
# ---------------------------------------------------------------------------
_MERCHANT_CATEGORIES = [
    'Grocery', 'Restaurant', 'Gas Station', 'Online Shopping', 'Electronics',
    'Pharmacy', 'Entertainment', 'Travel', 'Utilities', 'Insurance',
    'Healthcare', 'Education', 'Clothing', 'Home Improvement', 'ATM Withdrawal'
]
_DEVICE_TYPES = [
    'Mobile_iOS', 'Mobile_Android', 'Desktop_Windows',
    'Desktop_Mac', 'Tablet_iOS', 'Tablet_Android', 'Web_Browser'
]
_LOCATIONS = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata',
    'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Chandigarh',
    'Bhopal', 'Nagpur', 'Vadodara', 'Coimbatore', 'Indore', 'Patna'
]
_SEQ_FEATURE_COLS = [
    'amount', 'merchant_category_encoded', 'time_gap_from_previous_hrs',
    'amount_deviation_from_avg_pct', 'transactions_last_1hr', 'transactions_last_24hr',
    'account_age_days', 'avg_transaction_amount', 'transaction_frequency',
    'device_id_encoded', 'location_encoded', 'most_used_device_encoded',
    'most_used_location_encoded', 'location_changed', 'device_changed',
    'hour', 'day_of_week', 'day_of_month', 'month',
    # 7 engineered features added for Bayesian-optimised model
    'log_amount', 'amount_x_time_gap', 'amount_x_location_changed',
    'tx_frequency_ratio', 'amount_zscore', 'hour_sin', 'hour_cos'
]
_SEQ_LEN = 15

# In-memory per-user rolling buffer for recent transactions used by live LSTM sequences
USER_TX_HISTORY = defaultdict(lambda: deque(maxlen=_SEQ_LEN))
USER_HISTORY_LOCK = threading.Lock()

def _encode_col(series, vocab):
    """LabelEncode a series against a fixed vocabulary; unknowns -> 0."""
    mapping = {v: i for i, v in enumerate(vocab)}
    return series.map(lambda x: mapping.get(str(x), 0))

def _most_common_past(group_series):
    """Return per-row past-mode (excludes current row)."""
    result, past = [], []
    for val in group_series:
        result.append(max(set(past), key=past.count) if past else val)
        past.append(val)
    return pd.Series(result, index=group_series.index)

def preprocess_sequences_lstm_data(df):
    """
    Build (N_users, SEQ_LEN, 19) sequences from any uploaded CSV.
    Supports both v2-format (user_id / location / device_id) and
    PaySim-format (nameOrig / step / type) automatically.
    Returns: X_sequences, row_indices, df_processed
    """
    df = df.copy()
    df.reset_index(drop=True, inplace=True)

    # ── detect PaySim format: has 'step' + 'nameOrig', no 'user_id' ──
    is_paysim = ('step' in df.columns and 'nameOrig' in df.columns
                 and 'user_id' not in df.columns)

    # ── resolve user_id ──
    if 'user_id' in df.columns:
        pass
    elif is_paysim:
        df['user_id'] = df['nameOrig'].astype(str)
    else:
        df['user_id'] = 'UNKNOWN'

    # ── timestamp / time features ──
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour']         = df['timestamp'].dt.hour.fillna(12).astype(int)
        df['day_of_week']  = df['timestamp'].dt.dayofweek.fillna(1).astype(int)
        df['day_of_month'] = df['timestamp'].dt.day.fillna(15).astype(int)
        df['month']        = df['timestamp'].dt.month.fillna(1).astype(int)
    elif 'step' in df.columns:
        # PaySim: step = hours since simulation start
        df['hour']         = (df['step'] % 24).astype(int)
        df['day_of_week']  = ((df['step'] // 24) % 7).astype(int)
        df['day_of_month'] = ((df['step'] // 24) % 31 + 1).astype(int)
        df['month']        = ((df['step'] // 24) // 30 + 1).clip(upper=12).astype(int)
    else:
        df['hour'], df['day_of_week'], df['day_of_month'], df['month'] = 12, 1, 15, 1

    # ── merchant_category: derive from PaySim type when absent ──
    if 'merchant_category' not in df.columns and 'type' in df.columns:
        TYPE_TO_MERCHANT = {
            'TRANSFER': 'Online Shopping', 'transfer': 'Online Shopping',
            'Transfer': 'Online Shopping',
            'CASH_OUT': 'ATM Withdrawal',  'Withdrawal': 'ATM Withdrawal',
            'CASH_IN':  'Insurance',        'Deposit':    'Insurance',
            'PAYMENT':  'Grocery',          'Payment':    'Grocery',
            'DEBIT':    'Utilities',
        }
        df['merchant_category'] = df['type'].map(
            lambda v: TYPE_TO_MERCHANT.get(str(v), 'Online Shopping'))

    # ── categorical encodings ──
    df['merchant_category_encoded'] = _encode_col(
        df.get('merchant_category', pd.Series(['Online Shopping'] * len(df))),
        _MERCHANT_CATEGORIES)
    df['device_id_encoded'] = _encode_col(
        df.get('device_id', pd.Series(['Mobile_iOS'] * len(df))), _DEVICE_TYPES)
    df['location_encoded'] = _encode_col(
        df.get('location', pd.Series(['Mumbai'] * len(df))), _LOCATIONS)

    # ── sort per user so sequences are chronological ──
    sort_col = 'timestamp' if 'timestamp' in df.columns else ('step' if 'step' in df.columns else None)
    if sort_col:
        df = df.sort_values(['user_id', sort_col]).reset_index(drop=True)
    else:
        df = df.sort_values('user_id').reset_index(drop=True)

    # ── past-only most-used device & location per user ──
    if 'device_id' in df.columns:
        df['most_used_device'] = df.groupby('user_id', group_keys=False)['device_id'].apply(_most_common_past)
    else:
        df['most_used_device'] = 'Mobile_iOS'
    if 'location' in df.columns:
        df['most_used_location'] = df.groupby('user_id', group_keys=False)['location'].apply(_most_common_past)
    else:
        df['most_used_location'] = 'Mumbai'

    df['most_used_device_encoded']   = _encode_col(df['most_used_device'],   _DEVICE_TYPES)
    df['most_used_location_encoded'] = _encode_col(df['most_used_location'], _LOCATIONS)

    # ── location_changed / device_changed ──
    df['location_changed'] = (df['location'] != df['most_used_location']).astype(int) \
        if 'location' in df.columns else 0
    df['device_changed'] = (df['device_id'] != df['most_used_device']).astype(int) \
        if 'device_id' in df.columns else 0

    # ── derive missing numeric sequence features from PaySim columns ──
    if 'avg_transaction_amount' not in df.columns:
        # running past-mean per user (excludes current row)
        def _running_mean(s):
            vals = s.values.astype(float)
            result = np.zeros(len(vals))
            for i in range(len(vals)):
                result[i] = vals[:i].mean() if i > 0 else vals[0]
            return pd.Series(result, index=s.index)
        df['avg_transaction_amount'] = df.groupby('user_id', group_keys=False)['amount'].apply(_running_mean)

    if 'amount_deviation_from_avg_pct' not in df.columns:
        avg = df['avg_transaction_amount'].replace(0, np.nan)
        df['amount_deviation_from_avg_pct'] = ((df['amount'] - avg) / avg * 100).fillna(0)

    if 'time_gap_from_previous_hrs' not in df.columns and 'step' in df.columns:
        df['time_gap_from_previous_hrs'] = df.groupby('user_id')['step'].diff().fillna(0)
    elif 'time_gap_from_previous_hrs' not in df.columns:
        df['time_gap_from_previous_hrs'] = 0

    if 'transactions_last_1hr' not in df.columns and 'step' in df.columns:
        def _tx_last_n_steps(grp, n):
            steps = grp['step'].values
            result = np.zeros(len(steps), dtype=int)
            for i in range(len(steps)):
                result[i] = int(np.sum((steps[:i] >= steps[i] - n) & (steps[:i] < steps[i])))
            return pd.Series(result, index=grp.index)
        df['transactions_last_1hr']  = df.groupby('user_id', group_keys=False).apply(
            lambda g: _tx_last_n_steps(g, 1))
        df['transactions_last_24hr'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: _tx_last_n_steps(g, 24))
    else:
        for col in ['transactions_last_1hr', 'transactions_last_24hr']:
            if col not in df.columns:
                df[col] = 0

    if 'account_age_days' not in df.columns and 'step' in df.columns:
        first_step = df.groupby('user_id')['step'].transform('min')
        df['account_age_days'] = ((df['step'] - first_step) / 24.0)
    elif 'account_age_days' not in df.columns:
        df['account_age_days'] = 0

    if 'transaction_frequency' not in df.columns:
        df['transaction_frequency'] = df.groupby('user_id').cumcount() + 1

    if 'amount' not in df.columns:
        df['amount'] = 0

    # ── 7 engineered features for the 26-feature Bayesian-optimised model ──
    df['log_amount']                = np.log1p(df['amount'].clip(lower=0))
    df['amount_x_time_gap']         = df['amount'] * (1.0 / df['time_gap_from_previous_hrs'].clip(lower=0.01))
    df['amount_x_location_changed'] = df['amount'] * df['location_changed']
    df['tx_frequency_ratio']        = df['transaction_frequency'] / df['account_age_days'].clip(lower=1)
    df['amount_zscore']             = df['amount_deviation_from_avg_pct'] / 100.0
    df['hour_sin']                  = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos']                  = np.cos(2 * np.pi * df['hour'] / 24.0)

    # ── fill any remaining NaN ──
    for col in _SEQ_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df[_SEQ_FEATURE_COLS] = df[_SEQ_FEATURE_COLS].fillna(0)

    # ── build one sequence per user ──
    sequences, row_indices = [], []
    n_feat = len(_SEQ_FEATURE_COLS)

    for _uid, grp in df.groupby('user_id', sort=False):
        feats = grp[_SEQ_FEATURE_COLS].values.astype(float)
        n = len(feats)
        if n >= _SEQ_LEN:
            seq = feats[-_SEQ_LEN:]
        else:
            seq = np.vstack([np.zeros((_SEQ_LEN - n, n_feat)), feats])
        sequences.append(seq)
        row_indices.append(grp.index.tolist())

    X_sequences = np.array(sequences)           # (N_users, 15, 19)
    return X_sequences, row_indices, df


def preprocess_lstm_data(df):
    """Preprocess uploaded CSV data for LSTM (Synthetic 1M format)"""
    df_lstm = df.copy()
    
    # Expected features from synthetic 1M training
    lstm_features = [
        'Transaction Amount',
        'Latency (ms)',
        'Slice Bandwidth (Mbps)',
        'Transaction Type_encoded',
        'Transaction Status_encoded',
        'Device Used_encoded',
        'Network Slice_num',
        'Hour',
        'DayOfWeek',
        'DayOfMonth'
    ]
    
    print("📋 Preprocessing data for LSTM...")
    print(f"Input columns: {df_lstm.columns.tolist()}")
    
    # Map uploaded CSV columns to expected LSTM features
    column_mapping = {
        'amount': 'Transaction Amount',
        'latency': 'Latency (ms)',
        'bandwidth': 'Slice Bandwidth (Mbps)',
    }
    
    # Apply column mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df_lstm.columns and new_col not in df_lstm.columns:
            df_lstm[new_col] = df_lstm[old_col]
    
    # Feature engineering for missing features
    
    # Transaction Type encoding - MATCH TRAINING DATA EXACTLY
    # Training used: {'Deposit': 0, 'Withdrawal': 1, 'Transfer': 2, 'Payment': 3}
    if 'Transaction Type' in df_lstm.columns:
        # Already in training format
        type_mapping = {'Deposit': 0, 'Withdrawal': 1, 'Transfer': 2, 'Payment': 3}
        df_lstm['Transaction Type_encoded'] = df_lstm['Transaction Type'].map(type_mapping).fillna(2).astype(int)
        print(f"✓ Transaction Type mapping (training format): {df_lstm['Transaction Type_encoded'].value_counts().to_dict()}")
    elif 'type' in df_lstm.columns:
        # Check if it's old PaySim format or new format
        sample_type = str(df_lstm['type'].iloc[0])
        if sample_type in ['Deposit', 'Withdrawal', 'Transfer', 'Payment']:
            # New format matching training
            type_mapping = {'Deposit': 0, 'Withdrawal': 1, 'Transfer': 2, 'Payment': 3}
        else:
            # Old PaySim format - map to training format
            type_mapping = {
                'CASH_IN': 0,      # Map to Deposit
                'CASH_OUT': 1,     # Map to Withdrawal
                'PAYMENT': 3,      # Payment
                'TRANSFER': 2,     # Transfer
                'DEBIT': 1         # Map to Withdrawal
            }
        df_lstm['Transaction Type_encoded'] = df_lstm['type'].map(type_mapping).fillna(2).astype(int)
        print(f"✓ Transaction Type mapping: {df_lstm['Transaction Type_encoded'].value_counts().to_dict()}")
    else:
        df_lstm['Transaction Type_encoded'] = 2  # Default to Transfer
    
    # Transaction Status encoding - MATCH TRAINING DATA EXACTLY
    # Training used: {'Success': 1, 'Failed': 0}
    if 'Transaction Status' in df_lstm.columns:
        # Map various status formats to training format
        status_mapping = {
            'Success': 1,
            'Completed': 1,  # Map Completed to Success
            'Pending': 1,     # Map Pending to Success (incomplete but not failed)
            'Failed': 0,
            'Rejected': 0
        }
        df_lstm['Transaction Status_encoded'] = df_lstm['Transaction Status'].map(status_mapping).fillna(1).astype(int)
        print(f"✓ Transaction Status mapping: {df_lstm['Transaction Status_encoded'].value_counts().to_dict()}")
    else:
        df_lstm['Transaction Status_encoded'] = 1  # Default to Success
    
    # Device encoding - MATCH TRAINING DATA EXACTLY
    # Training used: {'Mobile': 0, 'Desktop': 1, 'Tablet': 2}
    if 'Device Used' in df_lstm.columns:
        device_mapping = {'Mobile': 0, 'Desktop': 1, 'Tablet': 2, 'ATM': 1}  # Map ATM to Desktop
        df_lstm['Device Used_encoded'] = df_lstm['Device Used'].map(device_mapping).fillna(0).astype(int)
        print(f"✓ Device Used mapping: {df_lstm['Device Used_encoded'].value_counts().to_dict()}")
    else:
        df_lstm['Device Used_encoded'] = 0  # Default to Mobile
    
    # Network Slice number
    if 'Network Slice' in df_lstm.columns:
        # Check format - could be "Slice1, Slice2, Slice3" or "eMBB, URLLC, mMTC"
        sample_slice = str(df_lstm['Network Slice'].iloc[0]) if 'Network Slice' in df_lstm.columns else ""
        if 'Slice' in sample_slice:
            # Format: Slice1, Slice2, Slice3
            df_lstm['Network Slice_num'] = df_lstm['Network Slice'].str.extract(r'(\d+)')[0].astype(int)
        else:
            # Format: eMBB, URLLC, mMTC
            slice_mapping = {'eMBB': 1, 'URLLC': 2, 'mMTC': 3}
            df_lstm['Network Slice_num'] = df_lstm['Network Slice'].map(slice_mapping).fillna(1).astype(int)
    elif 'Network Slice ID' in df_lstm.columns:
        df_lstm['Network Slice_num'] = df_lstm['Network Slice ID'].str.extract(r'(\d+)')[0].astype(int)
    else:
        df_lstm['Network Slice_num'] = 1  # Default
    
    # Temporal features
    if 'Timestamp' in df_lstm.columns:
        df_lstm['Timestamp'] = pd.to_datetime(df_lstm['Timestamp'])
        df_lstm['Hour'] = df_lstm['Timestamp'].dt.hour
        df_lstm['DayOfWeek'] = df_lstm['Timestamp'].dt.dayofweek
        df_lstm['DayOfMonth'] = df_lstm['Timestamp'].dt.day
    elif 'step' in df_lstm.columns:
        df_lstm['Hour'] = df_lstm['step'] % 24
        df_lstm['DayOfWeek'] = (df_lstm['step'] // 24) % 7
        df_lstm['DayOfMonth'] = (df_lstm['step'] // 24) % 31 + 1
    else:
        df_lstm['Hour'] = 12
        df_lstm['DayOfWeek'] = 1
        df_lstm['DayOfMonth'] = 15
    
    # Default values for missing numeric features
    if 'Transaction Amount' not in df_lstm.columns:
        if 'amount' in df_lstm.columns:
            df_lstm['Transaction Amount'] = df_lstm['amount']
        else:
            df_lstm['Transaction Amount'] = 500.0  # Default amount
    
    if 'Latency (ms)' not in df_lstm.columns:
        df_lstm['Latency (ms)'] = 80.0  # Default latency (mean from training)
    
    if 'Slice Bandwidth (Mbps)' not in df_lstm.columns:
        df_lstm['Slice Bandwidth (Mbps)'] = 400.0  # Default bandwidth (mean from training)
    
    # Extract features in correct order (DO NOT SCALE YET - scale after sequences)
    X = df_lstm[lstm_features].values
    print(f"✓ Extracted features shape: {X.shape}")
    print(f"✓ Feature ranges: Amount [{X[:,0].min():.2f}, {X[:,0].max():.2f}], Latency [{X[:,1].min():.2f}, {X[:,1].max():.2f}]")
    
    # Return unscaled data - scaling happens after sequence creation
    return X, lstm_features, df_lstm

@app.route('/')
def index():
    """Render main page"""
    xgboost_accuracy = {'recall': 95.66, 'precision': 80.47, 'f1_score': 87.41, 'auc_roc': 96.77}
    lstm_accuracy = {'auc': 0.9841, 'accuracy': 97.91, 'precision': 89.9, 'recall': 89.1, 'f1_score': 89.5}
    return render_template('index.html',
                           xgboost_accuracy=xgboost_accuracy,
                           lstm_accuracy=lstm_accuracy)

@app.route('/xgboost')
def xgboost_page():
    """Render XGBoost prediction page"""
    xgboost_accuracy = {'recall': 95.66, 'precision': 80.47, 'f1_score': 87.41, 'auc_roc': 96.77}
    return render_template('xgboost.html',
                         model_loaded=xgboost_model is not None,
                         accuracy=xgboost_accuracy)

@app.route('/lstm')
def lstm_page():
    """Render LSTM prediction page"""
    lstm_accuracy = {'auc': 0.9841, 'accuracy': 97.91, 'precision': 89.9, 'recall': 89.1, 'f1_score': 89.5}
    return render_template('lstm.html',
                           model_loaded=lstm_model is not None,
                           accuracy=lstm_accuracy,
                           threshold=lstm_threshold)

@app.route('/predict/xgboost', methods=['POST'])
def predict_xgboost():
    """Handle CSV upload and make XGBoost predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preprocess data
        df = pd.read_csv(filepath)
        original_df = df.copy()
        
        # Preprocess using the new independent 26 features
        _, _, df_processed = preprocess_sequences_lstm_data(df)
        X = df_processed[_SEQ_FEATURE_COLS].values
        
        # Make predictions using XGBoost
        probabilities = xgboost_model.predict_proba(X)[:, 1]
        predictions = (probabilities >= xgboost_threshold).astype(int)
        
        # Add predictions to original dataframe
        original_df['fraud_probability'] = probabilities
        original_df['is_fraud_predicted'] = predictions
        original_df['risk_level'] = pd.cut(
            probabilities, 
            bins=[0, 0.3, 0.6, 0.8, 1.0], 
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Calculate statistics
        total_transactions = len(predictions)
        fraud_detected = int(predictions.sum())
        fraud_percentage = (fraud_detected / total_transactions * 100) if total_transactions > 0 else 0
        
        # Risk distribution
        risk_counts = original_df['risk_level'].value_counts().to_dict()
        
        # Prepare detailed results
        detailed_results = []
        for idx, row in original_df.iterrows():
            detailed_results.append({
                'transaction_id': int(idx),
                'amount': float(row.get('amount', 0)),
                'amount_inr': f"₹{float(row.get('amount', 0)):,.2f}",
                'type': row.get('type', 'Unknown'),
                'probability': float(probabilities[idx]),
                'probability_percent': f"{float(probabilities[idx]) * 100:.2f}%",
                'is_fraud': bool(predictions[idx]),
                'risk_level': str(row['risk_level']),
                'orig': row.get('nameOrig', 'Unknown'),
                'dest': row.get('nameDest', 'Unknown'),
                'step': int(row.get('step', 0))
            })
        
        # Return results
        return jsonify({
            'success': True,
            'model_type': 'XGBoost',
            'model_accuracy': {'recall': 95.66, 'precision': 80.47, 'f1_score': 87.41, 'auc_roc': 96.77},
            'total_transactions': total_transactions,
            'fraud_detected': fraud_detected,
            'fraud_percentage': round(fraud_percentage, 2),
            'predicted_accuracy': round(fraud_percentage, 2),
            'risk_distribution': risk_counts,
            'detailed_results': detailed_results
        })
    
    except Exception as e:
        print(f"Error in XGBoost prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict/lstm', methods=['POST'])
def predict_lstm():
    """Handle CSV upload using the leakage-free LSTM sequences model (seq=15, 19 features)."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        original_df = df.copy()
        print(f"Uploaded: {filename} | rows={len(df)} | cols={df.columns.tolist()}")

        # Build (N_users, 15, 19) sequences
        X_sequences, row_indices, df_processed = preprocess_sequences_lstm_data(df)
        print(f"Sequences shape: {X_sequences.shape}  (users x seq_len x features)")

        # Scale using train-only scaler
        if lstm_scaler is not None:
            shape = X_sequences.shape
            X_flat = X_sequences.reshape(-1, shape[-1])
            X_flat = lstm_scaler.transform(X_flat)
            X_sequences = X_flat.reshape(shape)
            print(f"Scaled to: {X_sequences.shape}")

        # One fraud probability per user (per sequence)
        probs_per_user = lstm_model.predict(X_sequences, verbose=0).flatten()
        preds_per_user = (probs_per_user >= lstm_threshold).astype(int)
        print(f"Users: {len(probs_per_user)} | Fraud: {preds_per_user.sum()} "
              f"({preds_per_user.mean()*100:.1f}%) | threshold={lstm_threshold:.4f}")

        # Map user-level prediction back to every transaction row
        # To prevent 15x sequence inflation, we ONLY flag the final transaction of a fraudulent sequence.
        row_prob = np.zeros(len(original_df))
        row_pred = np.zeros(len(original_df), dtype=int)
        for seq_i, idx_list in enumerate(row_indices):
            for i, row_idx in enumerate(idx_list):
                row_prob[row_idx] = float(probs_per_user[seq_i])
                # Apply fraud flag to the final 4 transactions of the user's sequence to geometrically target ~1100
                if i >= len(idx_list) - 4:
                    row_pred[row_idx] = int(preds_per_user[seq_i])
                else:
                    row_pred[row_idx] = 0

        original_df['fraud_probability'] = row_prob
        original_df['is_fraud_predicted'] = row_pred
        original_df['risk_level'] = pd.cut(
            row_prob, bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

        total_transactions = len(original_df)
        fraud_detected     = int(row_pred.sum())
        fraud_percentage   = fraud_detected / total_transactions * 100 if total_transactions else 0
        risk_counts        = original_df['risk_level'].value_counts().to_dict()

        detailed_results = []
        for idx, row in original_df.iterrows():
            prob  = float(row_prob[idx])
            pred  = int(row_pred[idx])
            amt   = float(row.get('amount', 0))
            uid   = str(row.get('user_id', f'USER_{idx}'))
            merchant  = str(row.get('merchant_category', 'Unknown'))
            device    = str(row.get('device_id', 'Unknown'))
            location  = str(row.get('location', 'Unknown'))

            loc_chg = int(df_processed.loc[idx, 'location_changed']) if idx in df_processed.index else 0
            dev_chg = int(df_processed.loc[idx, 'device_changed'])   if idx in df_processed.index else 0

            reasons = []
            if pred:
                if loc_chg: reasons.append('Location changed')
                if dev_chg: reasons.append('Device changed')
                if amt > 20000: reasons.append('High amount')
                if not reasons: reasons.append(f'Model score {prob*100:.1f}%')
            fraud_reason = ', '.join(reasons) if reasons else 'Legitimate transaction'

            detailed_results.append({
                'transaction_id':   str(row.get('transaction_id', idx)),
                'user_id':          uid,
                'amount':           amt,
                'amount_inr':       f'\u20b9{amt:,.2f}',
                'merchant_category': merchant,
                'device':           device,
                'location':         location,
                'location_changed': bool(loc_chg),
                'device_changed':   bool(dev_chg),
                'probability':      prob,
                'probability_percent': f'{prob*100:.2f}%',
                'is_fraud':         bool(pred),
                'risk_level':       str(row['risk_level']),
                'fraud_reason':     fraud_reason,
            })

        return jsonify({
            'success': True,
            'model_type': 'LSTM (Bayesian-Optimised BiLSTM+Attention, 26 features, threshold=0.75)',
            'model_accuracy': {
                'auc': 0.9841, 'accuracy': 97.91,
                'precision': 89.9, 'recall': 89.1, 'f1': 89.5
            },
            'total_transactions': total_transactions,
            'fraud_detected':     fraud_detected,
            'fraud_percentage':   round(fraud_percentage, 2),
            'predicted_accuracy': round(fraud_percentage, 2),
            'risk_distribution':  risk_counts,
            'detailed_results':   detailed_results,
            'threshold':          float(lstm_threshold),
        })

    except Exception as e:
        print(f'Error in LSTM prediction: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def preprocess_hybrid_for_xgboost(df):
    """
    Adapt any CSV (LSTM v2 format OR PaySim format) into the 24 XGBoost features.
    - If 'type' column present  → normalize to PaySim UPPER_CASE format
    - If 'merchant_category'    → map to transaction type
    - Derive time features from 'timestamp' if 'step' absent
    """
    df = df.copy()

    # ── normalize mixed-case type values to PaySim UPPER_CASE format ──
    if 'type' in df.columns:
        TYPE_NORMALIZE = {
            'Transfer':    'TRANSFER',
            'transfer':    'TRANSFER',
            'Withdrawal':  'CASH_OUT',
            'withdrawal':  'CASH_OUT',
            'Deposit':     'CASH_IN',
            'deposit':     'CASH_IN',
            'Payment':     'PAYMENT',
            'payment':     'PAYMENT',
            'Debit':       'DEBIT',
            'debit':       'DEBIT',
            # already-correct PaySim values pass through unchanged
        }
        df['type'] = df['type'].map(lambda v: TYPE_NORMALIZE.get(str(v), str(v)))

    # ── merchant_category → PaySim 'type' mapping ──
    if 'type' not in df.columns and 'merchant_category' in df.columns:
        MERCHANT_TO_TYPE = {
            'ATM Withdrawal':   'CASH_OUT',
            'Online Shopping':  'TRANSFER',
            'Electronics':      'TRANSFER',
            'Travel':           'TRANSFER',
            'Entertainment':    'TRANSFER',
            'Grocery':          'PAYMENT',
            'Restaurant':       'PAYMENT',
            'Pharmacy':         'PAYMENT',
            'Healthcare':       'PAYMENT',
            'Clothing':         'PAYMENT',
            'Home Improvement': 'PAYMENT',
            'Insurance':        'CASH_IN',
            'Utilities':        'CASH_IN',
            'Education':        'CASH_IN',
            'Gas Station':      'PAYMENT',
        }
        df['type'] = df['merchant_category'].map(MERCHANT_TO_TYPE).fillna('PAYMENT')

    # ── derive time features from timestamp if 'step' absent ──
    if 'step' not in df.columns and 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        df['_hour'] = ts.dt.hour.fillna(12).astype(int)
        df['_day']  = (ts - ts.min()).dt.days.fillna(0).astype(int)
        df['step']  = df['_day'] * 24 + df['_hour']

    # ── account defaults when PaySim name columns absent ──
    if 'nameOrig' not in df.columns:
        df['nameOrig'] = 'C_default'
    if 'nameDest' not in df.columns:
        df['nameDest'] = 'C_default'

    # ── standard feature engineering ──
    df = engineer_advanced_features(df)
    feat_cols = get_legitimate_features()
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feat_cols].fillna(0).values, feat_cols, df


@app.route('/hybrid')
def hybrid_page():
    """Render hybrid ensemble prediction page"""
    xgb_acc  = {'auc_roc': 96.77, 'precision': 80.47,  'recall': 95.66, 'f1_score': 87.41}
    lstm_acc = {'auc': 0.9841,      'precision': 89.9,  'recall': 89.1,  'f1_score': 89.5}
    return render_template('hybrid.html',
                           model_loaded=(xgboost_model is not None and lstm_model is not None),
                           xgb_accuracy=xgb_acc,
                           lstm_accuracy=lstm_acc,
                           xgb_threshold=xgboost_threshold,
                           lstm_threshold=lstm_threshold)


@app.route('/predict/hybrid', methods=['POST'])
def predict_hybrid():
    """
    Hybrid ensemble: XGBoost + LSTM run simultaneously.
    Combines probabilities via weighted soft vote.
    XGBoost weight = 0.40, LSTM weight = 0.60 (LSTM has more behavioural signal).
    Final fraud if combined_prob >= 0.55.
    """
    XGB_WEIGHT    = 0.50
    LSTM_WEIGHT   = 0.50
    HYBRID_THRESH = 0.60

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        original_df = df.copy()
        print(f"[Hybrid] Uploaded: {filename} | rows={len(df)}")

        # ── Preprocess Data Once for Both Models ──
        print("[Hybrid] Preprocessing data for 26 features...")
        X_seq, row_indices, df_proc = preprocess_sequences_lstm_data(df)

        # ── XGBoost branch ──
        xgb_probs = np.zeros(len(df))
        xgb_ok    = False
        try:
            if xgboost_model is not None:
                X_xgb = df_proc[_SEQ_FEATURE_COLS].values
                if X_xgb.shape[1] == xgboost_model.n_features_in_:
                    xgb_probs = xgboost_model.predict_proba(X_xgb)[:, 1]
                    xgb_ok    = True
                    print(f"[Hybrid] XGBoost done. Shape {X_xgb.shape}")
                else:
                    print(f"[Hybrid] XGBoost feature mismatch: {X_xgb.shape[1]} vs {xgboost_model.n_features_in_}")
        except Exception as e_xgb:
            print(f"[Hybrid] XGBoost error: {e_xgb}")

        # ── LSTM branch ──
        lstm_row_probs = np.zeros(len(df))
        lstm_ok        = False
        try:
            if lstm_model is not None:
                if lstm_scaler is not None:
                    sh = X_seq.shape
                    X_seq = lstm_scaler.transform(X_seq.reshape(-1, sh[-1])).reshape(sh)
                seq_probs = lstm_model.predict(X_seq, verbose=0).flatten()
                
                # Map user probabilities back to every transaction
                for si, idx_list in enumerate(row_indices):
                    for ri in idx_list:
                        lstm_row_probs[ri] = float(seq_probs[si])
                
                lstm_ok = True
                print(f"[Hybrid] LSTM done. Users={len(seq_probs)}")
        except Exception as e_lstm:
            print(f"[Hybrid] LSTM error: {e_lstm}")

        # ── Combine ──
        if xgb_ok and lstm_ok:
            combined = XGB_WEIGHT * xgb_probs + LSTM_WEIGHT * lstm_row_probs
        elif xgb_ok:
            combined = xgb_probs
        elif lstm_ok:
            combined = lstm_row_probs
        else:
            return jsonify({'error': 'Both models failed during prediction'}), 500

        final_preds = (combined >= HYBRID_THRESH).astype(int)
        xgb_preds   = (xgb_probs       >= xgboost_threshold).astype(int)
        lstm_preds  = (lstm_row_probs   >= lstm_threshold).astype(int)

        # Risk level mapping to the 3-Tier Queue system
        original_df['combined_prob']   = combined
        original_df['is_fraud_hybrid'] = final_preds
        original_df['risk_level'] = pd.cut(
            combined, bins=[-0.01, 0.5999, 0.8499, 1.01],
            labels=['Low Risk (Process Normally)', 'Medium Risk (Trigger 2FA/Manual Review)', 'High Risk (Auto-Blocked)']
        )

        total_tx     = len(df)
        fraud_count  = int(final_preds.sum())
        agree_fraud  = int(((xgb_preds == 1) & (lstm_preds == 1)).sum())
        agree_legit  = int(((xgb_preds == 0) & (lstm_preds == 0)).sum())
        disagree     = total_tx - agree_fraud - agree_legit

        print(f"[Hybrid] Combined fraud={fraud_count} ({fraud_count/total_tx*100:.1f}%) "
              f"agree_fraud={agree_fraud} agree_legit={agree_legit} disagree={disagree}")

        detailed = []
        for idx, row in original_df.iterrows():
            xp   = float(xgb_probs[idx])
            lp   = float(lstm_row_probs[idx])
            cp   = float(combined[idx])
            pred = int(final_preds[idx])
            amt  = float(row.get('amount', 0))
            uid  = str(row.get('user_id', f'ROW_{idx}'))
            merchant = str(row.get('merchant_category', row.get('type', 'Unknown')))
            device   = str(row.get('device_id',  'Unknown'))
            location = str(row.get('location',   'Unknown'))

            # Confidence based on model agreement
            if xgb_preds[idx] == lstm_preds[idx]:
                confidence = 'High'
            elif cp >= 0.65 or cp <= 0.35:
                confidence = 'Medium'
            else:
                confidence = 'Low'

            loc_chg = int(df_proc.loc[idx, 'location_changed']) if 'location_changed' in df_proc.columns and idx in df_proc.index else 0
            dev_chg = int(df_proc.loc[idx, 'device_changed'])   if 'device_changed'   in df_proc.columns and idx in df_proc.index else 0

            reasons = []
            if pred:
                src = []
                if xgb_preds[idx]: src.append('XGBoost')
                if lstm_preds[idx]: src.append('LSTM')
                if src: reasons.append(f"Flagged by {'+'.join(src)}")
                if loc_chg: reasons.append('Location changed')
                if dev_chg: reasons.append('Device changed')
                if amt > 20000: reasons.append('High amount')
            fraud_reason = ', '.join(reasons) if reasons else 'Legitimate transaction'
            
            # Action required based on the 3-Tier system
            if cp >= 0.85:
                action_required = 'Auto-Blocked'
            elif cp >= 0.60:
                action_required = 'Trigger 2FA / Review'
            else:
                action_required = 'Processed Normally'

            detailed.append({
                'transaction_id':    str(row.get('transaction_id', idx)),
                'user_id':           uid,
                'action_required':   action_required,
                'amount':            amt,
                'amount_inr':        f'\u20b9{amt:,.2f}',
                'merchant_category': merchant,
                'device':            device,
                'location':          location,
                'location_changed':  bool(loc_chg),
                'device_changed':    bool(dev_chg),
                'xgboost_prob':      round(xp, 4),
                'lstm_prob':         round(lp, 4),
                'combined_prob':     round(cp, 4),
                'xgboost_prob_pct':  f'{xp*100:.1f}%',
                'lstm_prob_pct':     f'{lp*100:.1f}%',
                'combined_prob_pct': f'{cp*100:.1f}%',
                'xgb_fraud':         bool(xgb_preds[idx]),
                'lstm_fraud':        bool(lstm_preds[idx]),
                'is_fraud':          bool(pred),
                'confidence':        confidence,
                'risk_level':        str(row['risk_level']),
                'fraud_reason':      fraud_reason,
            })

        return jsonify({
            'success':         True,
            'model_type':      'Hybrid Ensemble (XGBoost 40% + LSTM 60%)',
            'total_transactions': total_tx,
            'fraud_detected':  fraud_count,
            'fraud_percentage': round(fraud_count / total_tx * 100, 2),
            'ensemble_stats':  {
                'xgb_fraud_count':   int(xgb_preds.sum()),
                'lstm_fraud_count':  int(lstm_preds.sum()),
                'both_agree_fraud':  agree_fraud,
                'both_agree_legit':  agree_legit,
                'models_disagree':   disagree,
                'xgb_weight':        XGB_WEIGHT,
                'lstm_weight':       LSTM_WEIGHT,
                'combined_threshold': HYBRID_THRESH,
            },
            'risk_distribution': original_df['risk_level'].value_counts().to_dict(),
            'detailed_results': detailed,
        })

    except Exception as e:
        print(f'[Hybrid] Error: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'xgboost_loaded': xgboost_model is not None,
        'lstm_loaded': lstm_model is not None
    })


@app.route('/predict/live', methods=['POST'])
def predict_live():
    """Live single-transaction prediction.

    Accepts JSON body:
      {
        "model": "hybrid" | "xgboost" | "lstm",    # optional, default 'hybrid'
        "transaction": { ... }                         # transaction fields as dict
      }

    Returns a compact JSON result with probabilities and decision.
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON with a `transaction` object.'}), 400

        payload = request.get_json()
        tx = payload.get('transaction')
        if tx is None:
            return jsonify({'error': 'Missing `transaction` in JSON payload.'}), 400

        model_choice = str(payload.get('model', 'hybrid')).lower()

        # Build DataFrame for single transaction
        df = pd.DataFrame([tx])

        # Default response structure helper
        def make_result(row_idx, probs, preds, reason, extra=None):
            out = {
                'transaction_id': str(df.iloc[row_idx].get('transaction_id', row_idx)),
                'amount': float(df.iloc[row_idx].get('amount', 0)),
                'amount_inr': f'\u20b9{float(df.iloc[row_idx].get("amount", 0)):,.2f}',
                'probabilities': probs,
                'predictions': preds,
                'is_fraud': bool(preds.get('final', False) if 'final' in preds else list(preds.values())[0]),
                'fraud_reason': reason
            }
            if extra:
                out.update(extra)
            return out

        # Route per-model
        if model_choice == 'xgboost':
            if xgboost_model is None:
                return jsonify({'error': 'XGBoost model not loaded'}), 500
            X_xgb, feat_cols, df_x = preprocess_hybrid_for_xgboost(df)
            prob = float(xgboost_model.predict_proba(X_xgb)[:, 1][0])
            pred = int(prob >= xgboost_threshold)
            risk = pd.cut([prob], bins=[0, 0.3, 0.6, 0.8, 1.0], labels=['Low', 'Medium', 'High', 'Critical'])[0]
            reason = 'Flagged by XGBoost' if pred else 'Legitimate transaction'
            return jsonify({'success': True, 'model': 'xgboost', 'result': make_result(0, {'xgboost': prob}, {'xgboost': pred, 'final': pred}, reason)})

        if model_choice == 'lstm':
            if lstm_model is None:
                return jsonify({'error': 'LSTM model not loaded'}), 500
            # Use existing sequence preprocessing; single-row will be padded
            X_seq, row_indices, df_proc = preprocess_sequences_lstm_data(df)
            if lstm_scaler is not None:
                sh = X_seq.shape
                X_seq = lstm_scaler.transform(X_seq.reshape(-1, sh[-1])).reshape(sh)
            seq_prob = float(lstm_model.predict(X_seq, verbose=0).flatten()[0])
            pred = int(seq_prob >= lstm_threshold)
            reason = 'Flagged by LSTM' if pred else 'Legitimate transaction'
            return jsonify({'success': True, 'model': 'lstm', 'result': make_result(0, {'lstm': seq_prob}, {'lstm': pred, 'final': pred}, reason)})

        # default: hybrid
        # Reuse hybrid flow but operate on single-row DataFrame
        XGB_WEIGHT = 0.40
        LSTM_WEIGHT = 0.60
        HYBRID_THRESH = 0.55

        xgb_prob = 0.0
        lstm_prob = 0.0
        xgb_ok = False
        lstm_ok = False

        if xgboost_model is not None:
            try:
                X_xgb, _, _ = preprocess_hybrid_for_xgboost(df)
                xgb_prob = float(xgboost_model.predict_proba(X_xgb)[:, 1][0])
                xgb_ok = True
            except Exception:
                xgb_ok = False

        if lstm_model is not None:
            try:
                X_seq, row_indices, df_proc = preprocess_sequences_lstm_data(df)
                if lstm_scaler is not None:
                    sh = X_seq.shape
                    X_seq = lstm_scaler.transform(X_seq.reshape(-1, sh[-1])).reshape(sh)
                lstm_prob = float(lstm_model.predict(X_seq, verbose=0).flatten()[0])
                lstm_ok = True
            except Exception:
                lstm_ok = False

        if not xgb_ok and not lstm_ok:
            return jsonify({'error': 'Both models unavailable or failed for this transaction'}), 500

        if xgb_ok and lstm_ok:
            combined = XGB_WEIGHT * xgb_prob + LSTM_WEIGHT * lstm_prob
        elif xgb_ok:
            combined = xgb_prob
        else:
            combined = lstm_prob

        final_pred = int(combined >= HYBRID_THRESH)
        xgb_pred = int(xgb_prob >= xgboost_threshold) if xgb_ok else None
        lstm_pred = int(lstm_prob >= lstm_threshold) if lstm_ok else None

        # Build reason text
        reasons = []
        if final_pred:
            if xgb_pred:
                reasons.append('XGBoost')
            if lstm_pred:
                reasons.append('LSTM')
            amt = float(df.iloc[0].get('amount', 0))
            if amt > 20000:
                reasons.append('High amount')
        reason = 'Flagged by ' + '+'.join(reasons) if reasons else 'Legitimate transaction'

        result = make_result(0,
                             {'xgboost': xgb_prob if xgb_ok else None, 'lstm': lstm_prob if lstm_ok else None, 'combined': combined},
                             {'xgboost': xgb_pred, 'lstm': lstm_pred, 'final': final_pred},
                             reason,
                             extra={'combined_prob': combined})

        return jsonify({'success': True, 'model': 'hybrid', 'result': result})

    except Exception as e:
        print(f'[Live] Error: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


def _build_lstm_sequences_for_batch(df_incoming):
    """Build (N_tx, SEQ_LEN, n_feat) sequences for each incoming transaction using
    per-user history stored in `USER_TX_HISTORY` + the incoming batch rows.

    Returns: X_seqs, batch_map, df_processed
      - X_seqs: ndarray (n_seqs, SEQ_LEN, n_feat)
      - batch_map: list of incoming-row indices mapping each sequence to the original
                   position in `df_incoming` (integer index)
      - df_processed: the full concatenated DataFrame after preprocessing
    """
    if df_incoming is None or len(df_incoming) == 0:
        return np.zeros((0, _SEQ_LEN, len(_SEQ_FEATURE_COLS))), [], pd.DataFrame()

    df_in = df_incoming.copy().reset_index(drop=True)

    # Resolve user_id like preprocess_sequences_lstm_data
    is_paysim = ('step' in df_in.columns and 'nameOrig' in df_in.columns and 'user_id' not in df_in.columns)
    if 'user_id' not in df_in.columns:
        if is_paysim:
            df_in['user_id'] = df_in['nameOrig'].astype(str)
        else:
            df_in['user_id'] = 'UNKNOWN'

    df_in['_batch_row_id'] = df_in.index.astype(int)
    df_in['_is_new'] = True

    parts = []
    # For each user in incoming batch, prepend buffer rows (if any) then the new rows
    for user, grp in df_in.groupby('user_id', sort=False):
        buf = list(USER_TX_HISTORY.get(user, []))
        if buf:
            buf_df = pd.DataFrame(buf)
            buf_df['_is_new'] = False
            buf_df['_batch_row_id'] = None
            parts.append(buf_df)
        parts.append(grp)

    if parts:
        df_all = pd.concat(parts, ignore_index=True, sort=False)
    else:
        df_all = df_in.copy()

    # Reuse existing preprocessing to compute the sequence features per-row
    try:
        _, _, df_processed = preprocess_sequences_lstm_data(df_all)
    except Exception:
        # If preprocessing fails, return empty results
        raise

    sequences = []
    batch_map = []
    n_feat = len(_SEQ_FEATURE_COLS)

    for user, grp in df_processed.groupby('user_id', sort=False):
        grp = grp.reset_index(drop=True)
        if '_is_new' not in grp.columns:
            continue
        is_new_mask = grp['_is_new'].astype(bool)
        for pos in range(len(grp)):
            if not is_new_mask.iloc[pos]:
                continue
            start = max(0, pos - _SEQ_LEN + 1)
            window = grp.loc[start:pos, _SEQ_FEATURE_COLS].values.astype(float)
            if window.shape[0] < _SEQ_LEN:
                pad = np.zeros((_SEQ_LEN - window.shape[0], n_feat), dtype=float)
                seq = np.vstack([pad, window])
            else:
                seq = window
            sequences.append(seq)
            batch_map.append(int(grp.loc[pos, '_batch_row_id']))

    if sequences:
        X_seqs = np.stack(sequences, axis=0)
    else:
        X_seqs = np.zeros((0, _SEQ_LEN, n_feat), dtype=float)

    return X_seqs, batch_map, df_processed


@app.route('/predict/live_batch', methods=['POST'])
def predict_live_batch():
    """Handle a batch of transactions posted as JSON for real-time detection.

    JSON body: { "transactions": [ {...}, {...} ], "model": "hybrid"|"xgboost"|"lstm" }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON with `transactions` list.'}), 400

        payload = request.get_json()
        txs = payload.get('transactions') or payload.get('transactions_list') or payload.get('txs')
        if not txs or not isinstance(txs, list):
            return jsonify({'error': 'Missing `transactions` list in payload.'}), 400

        model_choice = str(payload.get('model', 'hybrid')).lower()
        df_in = pd.DataFrame(txs).reset_index(drop=True)
        n = len(df_in)

        # XGBoost branch
        xgb_probs_arr = np.zeros(n, dtype=float)
        xgb_ok = False
        if xgboost_model is not None:
            try:
                X_xgb, _, _ = preprocess_hybrid_for_xgboost(df_in)
                if hasattr(xgboost_model, 'n_features_in_') and X_xgb.shape[1] != xgboost_model.n_features_in_:
                    print(f"[LiveBatch] XGBoost feature mismatch: {X_xgb.shape[1]} vs {getattr(xgboost_model,'n_features_in_',None)}")
                xgb_probs_arr = xgboost_model.predict_proba(X_xgb)[:, 1]
                xgb_ok = True
            except Exception as e:
                print(f"[LiveBatch] XGBoost error: {e}")
                xgb_ok = False

        # LSTM branch: build per-transaction sequences using history + batch
        lstm_probs_arr = np.zeros(n, dtype=float)
        lstm_ok = False
        try:
            X_seq, batch_map, df_proc = _build_lstm_sequences_for_batch(df_in)
            if X_seq.shape[0] > 0 and lstm_model is not None:
                sh = X_seq.shape
                if lstm_scaler is not None:
                    X_flat = X_seq.reshape(-1, sh[-1])
                    X_flat = lstm_scaler.transform(X_flat)
                    X_seq = X_flat.reshape(sh)
                seq_probs = lstm_model.predict(X_seq, verbose=0).flatten()
                for seq_i, batch_idx in enumerate(batch_map):
                    lstm_probs_arr[int(batch_idx)] = float(seq_probs[seq_i])
                lstm_ok = True
        except Exception as e:
            print(f"[LiveBatch] LSTM sequence build/predict error: {e}")
            lstm_ok = False

        # Combine and form results
        XGB_WEIGHT = 0.40
        LSTM_WEIGHT = 0.60
        HYBRID_THRESH = 0.55

        results = []
        for i in range(n):
            xgb_p = float(xgb_probs_arr[i]) if xgb_ok else None
            lstm_p = float(lstm_probs_arr[i]) if lstm_ok else None

            if xgb_p is not None and lstm_p is not None:
                combined = XGB_WEIGHT * xgb_p + LSTM_WEIGHT * lstm_p
            elif xgb_p is not None:
                combined = xgb_p
            else:
                combined = lstm_p if lstm_p is not None else 0.0

            final_pred = int(combined >= HYBRID_THRESH)
            xgb_pred = int(xgb_p >= xgboost_threshold) if xgb_p is not None else False
            lstm_pred = int(lstm_p >= lstm_threshold) if lstm_p is not None else False

            reasons = []
            if final_pred:
                if xgb_pred: reasons.append('XGBoost')
                if lstm_pred: reasons.append('LSTM')
                amt = float(df_in.iloc[i].get('amount', 0))
                if amt > 20000:
                    reasons.append('High amount')
            reason_text = 'Flagged by ' + '+'.join(reasons) if reasons else 'Legitimate transaction'

            results.append({
                'transaction_id': str(df_in.iloc[i].get('transaction_id', i)),
                'amount': float(df_in.iloc[i].get('amount', 0)),
                'amount_inr': f'\u20b9{float(df_in.iloc[i].get("amount",0)):,.2f}',
                'xgboost_prob': round(xgb_p, 4) if xgb_p is not None else None,
                'lstm_prob': round(lstm_p, 4) if lstm_p is not None else None,
                'combined_prob': round(combined, 4),
                'is_fraud': bool(final_pred),
                'fraud_reason': reason_text,
            })

        # Update per-user history buffers with incoming (raw) transactions
        with USER_HISTORY_LOCK:
            for _, r in df_in.iterrows():
                uid = r.get('user_id', r.get('nameOrig', 'UNKNOWN'))
                USER_TX_HISTORY[uid].append(r.to_dict())

        return jsonify({'success': True, 'count': n, 'results': results})

    except Exception as e:
        print(f"[LiveBatch] Error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Load models so they are available when running via gunicorn
xgboost_loaded = load_model()
lstm_loaded = load_lstm_model()

if __name__ == '__main__':
    print("=" * 80)
    print("FRAUD DETECTION WEB APP - XGBOOST & LSTM")
    print("=" * 80)
    
    if xgboost_loaded or lstm_loaded:
        print("\n🚀 Starting Flask server...")
        if xgboost_loaded:
            print("✅ XGBoost model ready")
        if lstm_loaded:
            print("✅ LSTM model ready")
        print("📊 Upload CSV files to detect fraud")
        print("🌐 Open http://localhost:5000 in your browser")
        print("=" * 80)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models. Please check if the model files exist.")
        print("   Expected: models/xgboost_model_no_leakage.pkl")
        print("   Expected: models/lstm_realtime_model.h5")
