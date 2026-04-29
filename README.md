# Fraud Detection System

Advanced Machine Learning & Deep Learning system for financial fraud detection using XGBoost and LSTM models.

## 🏗️ Project Structure

```
Major_project/
│
├── 📁 Data Loading & Preprocessing
│   ├── 01_data_loading_eda.py              # Load and explore dataset
│   ├── 02_feature_engineering.py           # Create engineered features
│   ├── 03_preprocessing_no_leakage.py      # Data preprocessing without leakage
│   └── Synthetic_Financial_datasets_log.csv # Raw dataset (6.3M transactions)
│
├── 📁 Model Training
│   ├── 04_xgboost_model_no_leakage.py      # XGBoost model training
│   ├── 05_lstm_model_no_leakage.py         # LSTM model training
│   ├── 06_lstm_realtime_model.py           # Real-time LSTM model
│   ├── 07_hybrid_xgboost_lstm_model.py     # Hybrid ensemble model
│   └── 08_meta_model_hybrid.py             # Meta-learning model
│
├── 📁 Web Application
│   ├── app.py                              # Flask backend server
│   ├── templates/
│   │   ├── index.html                      # Home page (model selection)
│   │   ├── xgboost.html                    # XGBoost prediction interface
│   │   ├── lstm.html                       # LSTM prediction interface
│   │   └── hybrid.html                     # Hybrid XGBoost+LSTM interface
│   └── uploads/                            # Temporary file uploads
│
├── 📁 Models & Artifacts
│   ├── models/
│   │   ├── xgboost_model_no_leakage.pkl    # Trained XGBoost model
│   │   ├── lstm_realtime_model.h5          # Trained LSTM model
│   │   ├── lstm_realtime_scaler.pkl        # LSTM feature scaler
│   │   ├── *_report.txt                    # Model performance reports
│   │   └── *_history.pkl                   # Training history
│   └── plots/                              # Training visualizations
│
├── 📁 Processed Data
│   ├── engineered_data.csv                 # Feature-engineered dataset
│   ├── preprocessed_data_no_leakage/       # Clean train/test splits
│   └── sample_dataset_10k.csv              # Sample dataset for testing
│
├── 📁 Utilities
│   ├── create_sample_dataset.py            # Generate sample test data
│   ├── feature_list.txt                    # List of features
│   ├── preprocessing_summary.txt           # Preprocessing details
│   └── xgboost_report_no_leakage.txt       # XGBoost results
│
├── requirements.txt                        # Python dependencies
├── install_packages.bat                    # Windows installer
└── README.md                               # This file
```

## 🎯 Model Performance

### ⚡ XGBoost Model
Fast and accurate gradient boosting model optimized for fraud detection with high precision.
- **Recall**: 95.66%
- **Precision**: 80.47%
- **PR AUC**: 96.77%
- **F1-Score**: 87.41%
- **Features**: 24 Engineered Features
- **Key Traits**: 78.49% Fraud Detection Rate, Fast Training & Prediction, Excellent for Tabular Data

### 🧠 LSTM Model
Deep learning recurrent neural network for sequential pattern recognition in transaction data.
- **Recall**: 89.10%
- **Precision**: 89.90%
- **PR AUC**: 98.41%
- **F1-Score**: 89.50%
- **Features**: 19 Optimized Features
- **Key Traits**: 89.10% Fraud Detection Rate, Sequential Pattern Learning, Trained on 3.9M Transactions

### ⚡ Hybrid Ensemble
XGBoost (40%) + LSTM (60%) soft-vote ensemble for maximum fraud detection accuracy.
- **Combined Recall**: 98.72%
- **LSTM/XGB Weight**: 60/40
- **Ensemble Threshold**: 0.55
- **Vote Method**: Soft Output
- **Key Traits**: Both models run simultaneously, Per-row XGBoost + LSTM probability, Confidence indicator per transaction, Model agreement breakdown

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web Application
```bash
python app.py
```

### 3. Open Browser
Navigate to: `http://localhost:5000`

## 📊 Features

### Web Interface
- ✅ Upload CSV files for fraud detection (up to 100MB)
- ✅ Choose between XGBoost, LSTM, or Hybrid ensemble models
- ✅ View predictions in Indian Rupees (₹)
- ✅ Paginated transaction lists (50 per page)
- ✅ Professional banking color scheme
- ✅ Real-time fraud probability scores
- ✅ Risk level classification (Low/Medium/High/Critical)
- ✅ Model accuracy metrics display

### Data Processing
- No data leakage (balance features excluded)
- Proper train/test splits
- SMOTE balancing for imbalanced classes
- 29 legitimate engineered features
- Robust scaling

## 📁 Sample Datasets

### Generate Sample Data
```bash
python create_sample_dataset.py
```
Creates `sample_dataset_10k.csv` with 10,000 transactions for testing.

## 🔧 Training New Models

### Train XGBoost
```bash
python 04_xgboost_model_no_leakage.py
```

### Train LSTM
```bash
python 06_lstm_realtime_model.py
```

## 📝 Notes

- All models use leak-free features suitable for real-time deployment
- Fraud detection prioritized over precision (financial security focus)
- Models trained on 100,000 transaction sample for efficiency
- Full dataset: 6.3M transactions available

## 🛡️ Security

- No sensitive balance information used in predictions
- Production-ready fraud detection thresholds
- Optimized for real-time transaction screening

## 📫 Support

For issues or questions, refer to model reports in the `models/` directory.
