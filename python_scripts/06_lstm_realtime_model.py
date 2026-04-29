"""
06_lstm_realtime_model.py
Trains a lightweight LSTM for real-time (low-latency) inference.
Uses a smaller architecture compared to the full sequences model.
"""

import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

LSTM_TRAINING_DIR = '../lstm_training_data'
MODELS_DIR        = '../models'
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_realtime_model(seq_len, n_feats):
    """Smaller/faster architecture for real-time use."""
    inp = keras.Input(shape=(seq_len, n_feats))
    x   = layers.LSTM(32, return_sequences=False, dropout=0.2)(inp)
    x   = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1,  activation='sigmoid')(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-3),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    return model


if __name__ == '__main__':
    # Load preprocessed sequences from lstm_training_data/
    X = np.load(f'{LSTM_TRAINING_DIR}/X_train_lstm.npy')
    y = np.load(f'{LSTM_TRAINING_DIR}/y_train_lstm.npy')
    print(f"Train sequences: {X.shape}  Fraud: {y.sum()}/{len(y)}")

    sh     = X.shape
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X.reshape(-1, sh[-1])).reshape(sh)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sc, y, test_size=0.15, random_state=SEED, stratify=y
    )
    fraud_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = build_realtime_model(sh[1], sh[2])
    model.summary()

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=25, batch_size=512,
        class_weight={0: 1.0, 1: fraud_w},
        callbacks=[EarlyStopping(monitor='val_AUC', patience=7,
                                  restore_best_weights=True, mode='max')],
        verbose=1
    )

    X_test = np.load(f'{LSTM_TRAINING_DIR}/X_test_lstm.npy')
    y_test = np.load(f'{LSTM_TRAINING_DIR}/y_test_lstm.npy')
    X_test_sc = scaler.transform(X_test.reshape(-1, sh[-1])).reshape(X_test.shape)

    y_prob  = model.predict(X_test_sc, verbose=0).flatten()
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nRealtime LSTM  ROC-AUC={roc_auc:.4f}")

    model.save(f'{MODELS_DIR}/lstm_realtime_model.h5')
    with open(f'{MODELS_DIR}/lstm_realtime_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Saved lstm_realtime_model.h5 and lstm_realtime_scaler.pkl")
