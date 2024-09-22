import pandas as pd
import numpy as np
from datetime import datetime
from cryptography.fernet import Fernet
from crypto_analysis import perform_kyber_operations
from joblib import load  # For loading Isolation Forest
import tensorflow as tf  # For loading LSTM

# Generate a key and cipher suite
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def simulate_swift_transactions(n_transactions=1000):
    """
    Simulate SWIFT transactions and integrate CRYSTALS-Kyber operations.
    """
    transactions = []
    for _ in range(n_transactions):
        transaction = {
            'transaction_id': np.random.randint(100000, 999999),
            'sender': f'Sender_{np.random.randint(1, 100)}',
            'receiver': f'Receiver_{np.random.randint(1, 100)}',
            'amount': np.random.uniform(1000, 100000),
            'currency': np.random.choice(['USD', 'EUR', 'GBP']),
            'timestamp': datetime.now(),
            'sender_encrypted': cipher_suite.encrypt(f'Sender_{np.random.randint(1, 100)}'.encode()),
            'receiver_encrypted': cipher_suite.encrypt(f'Receiver_{np.random.randint(1, 100)}'.encode()),
            'amount_encrypted': cipher_suite.encrypt(str(np.random.uniform(1000, 100000)).encode())
        }

        # Integrate CRYSTALS-Kyber operations
        kyber_data = perform_kyber_operations()
        transaction.update(kyber_data)

        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

# Function for model monitoring (Isolation Forest & LSTM separately)
def monitor_transactions_isolation_forest(transactions, isolation_forest_model):
    features = transactions[['amount', 'public_key', 'ciphertext']]  # example features
    isolation_forest_preds = isolation_forest_model.predict(features)
    transactions['isolation_forest_anomaly'] = isolation_forest_preds
    return transactions

def monitor_transactions_lstm(transactions, lstm_model):
    features = transactions[['amount', 'public_key', 'ciphertext']]  # example features
    # Prepare LSTM input
    X_seq, _ = prepare_sequences(features.values, sequence_length=5)  
    lstm_preds = lstm_model.predict(X_seq)
    transactions['lstm_anomaly'] = lstm_preds
    return transactions

# Function for preparing sequences for LSTM
def prepare_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# New function for ensemble prediction
def ensemble_model(isolation_forest_model, lstm_model, X_test):
    # Get predictions from Isolation Forest
    isolation_forest_preds = isolation_forest_model.predict(X_test)
    
    # Prepare LSTM data
    X_seq, _ = prepare_sequences(X_test, sequence_length=5)
    
    # Get predictions from LSTM
    lstm_preds = lstm_model.predict(X_seq)
    
    # Ensemble decision (soft voting by averaging)
    ensemble_preds = (isolation_forest_preds + lstm_preds) / 2
    return ensemble_preds

if __name__ == "__main__":
    # Simulate SWIFT transactions
    transactions = simulate_swift_transactions(n_transactions=1000)
    print("First few simulated transactions:")
    print(transactions.head())

    # Load the pre-trained models
    isolation_forest_model = load('isolation_forest_model.joblib')
    lstm_model = tf.keras.models.load_model('lstm_model.h5')

    # Step 1: Run Isolation Forest predictions
    transactions_with_if_results = monitor_transactions_isolation_forest(transactions, isolation_forest_model)
    print("Transactions with Isolation Forest anomaly predictions:")
    print(transactions_with_if_results[['transaction_id', 'isolation_forest_anomaly']].head())

    # Step 2: Run LSTM predictions
    transactions_with_lstm_results = monitor_transactions_lstm(transactions, lstm_model)
    print("Transactions with LSTM anomaly predictions:")
    print(transactions_with_lstm_results[['transaction_id', 'lstm_anomaly']].head())

    # Step 3: Combine predictions with ensemble logic (after both models ran)
    features = transactions[['amount', 'public_key', 'ciphertext']].values
    ensemble_predictions = ensemble_model(isolation_forest_model, lstm_model, features)
    transactions['ensemble_anomaly'] = ensemble_predictions
    print("Transactions with ensemble anomaly predictions:")
    print(transactions[['transaction_id', 'ensemble_anomaly']].head())

    print("Anomaly detection complete.")
