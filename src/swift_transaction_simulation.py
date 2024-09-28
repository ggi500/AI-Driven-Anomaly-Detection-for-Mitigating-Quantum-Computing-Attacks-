import os 
import numpy as np
import pandas as pd
from datetime import datetime
from cryptography.fernet import Fernet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from joblib import load  # For loading Isolation Forest
import tensorflow as tf  # For loading LSTM
from crypto_analysis import perform_kyber_operations
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine  # Opacus for differential privacy in training

# Create a Faker instance
fake = Faker()

# Generate a key and cipher suite
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Visualization functions
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Anomaly', 'Anomaly'], yticklabels=['No Anomaly', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig('ensemble_confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('ensemble_roc_curve.png')
    plt.close()

# Load datasets 
def load_ieee_cis_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} does not exist.")
        return None
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    return df

def load_paysim_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} does not exist.")
        return None
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    return df

# Step 1: Check and Sort Data for Time-Series Structure
def check_time_series_structure(df):
    if 'timestamp' not in df.columns:
        raise ValueError("Dataset must contain a 'timestamp' column for LSTM sequence preparation.")
    df = df.sort_values(by='timestamp')  # Ensure the data is sorted by time
    return df

# Data Preprocessing 
def preprocess_data(df):
    if df is None:
        return None
    df = clean_data(df)
    df = normalize_data(df)
    return df

def clean_data(df):
    df = df.drop_duplicates()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())
    return df

# Step 2: Preprocess the Data Using MinMaxScaler or StandardScaler
def normalize_data(df, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization method. Choose 'minmax' or 'zscore'.")
    
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

# Attack-related feature engineering
def engineer_attack_features(df):
    df['failed_decapsulation_attempts'] = np.random.randint(0, 5, size=len(df))
    df['repeated_access_attempts'] = np.random.randint(0, 10, size=len(df))
    df['network_traffic_burst'] = np.random.choice([0, 1], size=len(df))  # Binary feature indicating traffic spike
    return df

# Cryptographic and Attack Feature Engineering
def engineer_features(df):
    df['qr_key_size'] = np.random.choice([3328, 4096, 6528], size=len(df))
    df['qr_signature_length'] = np.random.randint(2000, 4000, size=len(df))
    df['qr_encapsulation_time'] = np.random.uniform(0.001, 0.005, size=len(df))
    df['qr_decapsulation_time'] = np.random.uniform(0.001, 0.005, size=len(df))
    
    df = engineer_attack_features(df)
    return df

# Prepare sequences for LSTM input
def prepare_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# New validation function
def check_against_swift_standard(swift_data):
    expected_fields = ['transaction_id', 'sender_bic', 'receiver_bic', 'amount', 'currency', 'timestamp', 'ordering_customer', 'beneficiary_customer', 'transaction_reference']
    actual_fields = swift_data.columns.tolist()
    missing_fields = [field for field in expected_fields if field not in actual_fields]
    if missing_fields:
        print(f"\nWarning: The following expected fields are missing from the dataset: {missing_fields}")
    else:
        print("\nDataset structure matches expected SWIFT-like transaction fields.")
    return swift_data

# Simulate SWIFT transactions and integrate CRYSTALS-Kyber operations
def simulate_swift_transactions(n_transactions=1000):
    transactions = []
    for _ in range(n_transactions):
        transaction = {
            'transaction_id': np.random.randint(100000, 999999),
            'sender_bic': f"BIC-{np.random.randint(1000, 9999)}",  # Simulating BIC (SWIFT code)
            'receiver_bic': f"BIC-{np.random.randint(1000, 9999)}",
            'amount': np.random.uniform(1000, 100000),
            'currency': np.random.choice(['USD', 'EUR', 'GBP']),
            'timestamp': datetime.now(),
            'ordering_customer': fake.name(),
            'beneficiary_customer': fake.name(),
            'sender_encrypted': cipher_suite.encrypt(f'Sender_{np.random.randint(1, 100)}'.encode()),
            'receiver_encrypted': cipher_suite.encrypt(f'Receiver_{np.random.randint(1, 100)}'.encode()),
            'amount_encrypted': cipher_suite.encrypt(str(np.random.uniform(1000, 100000)).encode()),
            'transaction_reference': f"REF-{np.random.randint(100000, 999999)}",  # Simulating a reference number
        }

        # Perform Kyber operations
        kyber_data = perform_kyber_operations()
        transaction.update(kyber_data)

        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    
    # Add the check here
    df = check_against_swift_standard(df)
    
    return df

# Model monitoring with Isolation Forest
def monitor_transactions_isolation_forest(transactions, isolation_forest_model):
    features = transactions[['amount', 'public_key', 'ciphertext']]  # example features
    isolation_forest_preds = isolation_forest_model.predict(features)
    transactions['isolation_forest_anomaly'] = isolation_forest_preds
    return transactions

# Model monitoring with LSTM
def monitor_transactions_lstm(transactions, lstm_model):
    features = transactions[['amount', 'public_key', 'ciphertext']]  # example features
    X_seq, _ = prepare_sequences(features.values, sequence_length=5)
    lstm_preds = lstm_model.predict(X_seq)
    transactions['lstm_anomaly'] = lstm_preds
    return transactions

# Ensemble prediction function
def ensemble_model(isolation_forest_model, lstm_model, X_test):
    isolation_forest_preds = isolation_forest_model.predict(X_test)
    X_seq, _ = prepare_sequences(X_test, sequence_length=5)
    lstm_preds = lstm_model.predict(X_seq)
    ensemble_preds = (isolation_forest_preds + lstm_preds) / 2
    return ensemble_preds

# SWIFT transformation functions
def generate_bic_code():
    """Generate a random BIC (Bank Identifier Code) for SWIFT messages."""
    return f"BIC-{np.random.randint(1000, 9999)}"

def transform_ieee_to_swift(df):
    df['amount'] = df['TransactionAmt']
    df['currency'] = np.random.choice(['USD', 'EUR', 'GBP', 'JPY', 'CHF'], size=len(df))
    df['timestamp'] = pd.to_datetime(df['TransactionDT'], unit='s')
    df['sender_bic'] = df['card1'].apply(lambda x: generate_bic_code())  # Using card1 as sender identifier
    df['receiver_bic'] = df['card2'].apply(lambda x: generate_bic_code())  # Using card2 as receiver identifier
    df['transaction_reference'] = [f"REF-{np.random.randint(100000, 999999)}" for _ in range(len(df))]
    df['ordering_customer'] = df['card1'].apply(lambda x: fake.name())
    df['beneficiary_customer'] = df['card2'].apply(lambda x: fake.name())
    return df

def transform_unsw_to_swift(df):
    df['amount'] = np.random.uniform(1000, 100000, size=len(df))
    df['currency'] = np.random.choice(['USD', 'EUR', 'GBP', 'JPY', 'CHF'], size=len(df))
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['sender_bic'] = df['src_ip'].apply(lambda x: generate_bic_code())  # Using src_ip as sender identifier
    df['receiver_bic'] = df['dst_ip'].apply(lambda x: generate_bic_code())  # Using dst_ip as receiver identifier
    df['transaction_reference'] = [f"REF-{np.random.randint(100000, 999999)}" for _ in range(len(df))]
    df['ordering_customer'] = df['src_ip'].apply(lambda x: fake.name())
    df['beneficiary_customer'] = df['dst_ip'].apply(lambda x: fake.name())
    return df

def transform_paysim_to_swift(df):
    df['currency'] = np.random.choice(['USD', 'EUR', 'GBP', 'JPY', 'CHF'], size=len(df))
    df['sender_bic'] = df['nameOrig'].apply(lambda x: generate_bic_code())  # Using nameOrig as sender identifier
    df['receiver_bic'] = df['nameDest'].apply(lambda x: generate_bic_code())  # Using nameDest as receiver identifier
    df['transaction_reference'] = [f"REF-{np.random.randint(100000, 999999)}" for _ in range(len(df))]
    df['ordering_customer'] = df['nameOrig'].apply(lambda x: fake.name())
    df['beneficiary_customer'] = df['nameDest'].apply(lambda x: fake.name())
    return df

def transform_cic_to_swift(df):
    df['amount'] = df['Total Fwd Packets'].apply(lambda x: np.random.uniform(1000, 5000) * (x / 100))
    df['currency'] = np.random.choice(['USD', 'EUR', 'GBP', 'JPY', 'CHF'], size=len(df))
    df['timestamp'] = pd.to_datetime(df['Flow Duration'], unit='ms', errors='coerce')
    df['sender_bic'] = df['Source IP'].apply(lambda x: generate_bic_code())  # Using Source IP as sender identifier
    df['receiver_bic'] = df['Destination IP'].apply(lambda x: generate_bic_code())  # Using Destination IP as receiver identifier
    df['transaction_reference'] = [f"REF-{np.random.randint(100000, 999999)}" for _ in range(len(df))]
    df['ordering_customer'] = df['Source IP'].apply(lambda x: fake.name())
    df['beneficiary_customer'] = df['Destination IP'].apply(lambda x: fake.name())
    return df

# Main function to execute the flow
if __name__ == "__main__":
    # Load datasets 
    ieee_cis_data = load_ieee_cis_data('Data/IEEE-CIS-Fraud-Detection-master/train_transaction.csv')
    paysim_data = load_paysim_data('Data/PaySim-master/PS_20174392719_1491204439457_log.csv')

    # Preprocess datasets 
    ieee_cis_data = preprocess_data(ieee_cis_data)
    paysim_data = preprocess_data(paysim_data)

    # Step 1: Check and Sort the Data for Time-Series Structure
    ieee_cis_data = check_time_series_structure(ieee_cis_data)
    paysim_data = check_time_series_structure(paysim_data)

    # Engineer cryptographic and attack-related features
    ieee_cis_data = engineer_features(ieee_cis_data)
    paysim_data = engineer_features(paysim_data)

    # Combine data and sort by timestamp for LSTM
    combined_data = pd.concat([ieee_cis_data, paysim_data], axis=0)

    # Step 2: Preprocess the combined dataset
    combined_data = normalize_data(combined_data)

    # Simulate SWIFT transactions
    simulated_transactions = simulate_swift_transactions(n_transactions=1000)
    print("Simulated SWIFT transactions:")
    print(simulated_transactions.head())

    # Select features for LSTM
    X = combined_data[['amount', 'qr_key_size', 'qr_encapsulation_time', 'qr_decapsulation_time',
                       'failed_decapsulation_attempts', 'repeated_access_attempts', 'network_traffic_burst']].values
    y = combined_data['is_fraud'].values

    # Prepare sequences for LSTM
    sequence_length = 10  # Adjust based on the required sequence length
    X_seq, y_seq = prepare_sequences(X, sequence_length)

    # Load models
    isolation_forest_model, lstm_model = load('Models/trained_models/trained_isolation_forest_model.pkl'), \
                                         tf.keras.models.load_model('Models/trained_models/trained_lstm_model.h5')

    # Make predictions
    try:
        isolation_forest_preds = isolation_forest_model.predict(X)
        lstm_preds = lstm_model.predict(X_seq)
    except Exception as e:
        print(f"Error during predictions: {e}")
        exit(1)

    # Ensemble predictions
    ensemble_preds = ensemble_model(isolation_forest_model, lstm_model, X)

    # Visualize confusion matrix and ROC curve
    y_true = combined_data['is_fraud']
    y_pred = np.round(ensemble_preds)
    
    plot_confusion_matrix(y_true, y_pred, title="Ensemble Confusion Matrix")
    plot_roc_curve(y_true, ensemble_preds, title="Ensemble ROC Curve")

    print("Confusion matrix and ROC curve saved as images.")
