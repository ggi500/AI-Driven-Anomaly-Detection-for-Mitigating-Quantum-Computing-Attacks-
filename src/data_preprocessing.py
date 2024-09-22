import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pqcrypto.kem import kyber512
from faker import Faker
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine  # Opacus for differential privacy in training
from crypto_analysis import analyze_key_sizes, analyze_encapsulation_times, analyze_decapsulation_times

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
    """
    Engineer features related to cryptographic attack behaviors.
    
    Parameters:
    - df: pandas DataFrame, the input dataset.

    Returns:
    - df: pandas DataFrame, with new engineered attack-related features.
    """
    df['failed_decapsulation_attempts'] = np.random.randint(0, 5, size=len(df))
    df['repeated_access_attempts'] = np.random.randint(0, 10, size=len(df))
    df['network_traffic_burst'] = np.random.choice([0, 1], size=len(df))  # Binary feature indicating traffic spike
    return df

# Cryptographic and Attack Feature Engineering
def engineer_features(df):
    """
    Engineer features for quantum-resistant cryptographic operations and potential cryptographic attacks.
    
    Parameters:
    - df: pandas DataFrame, the input dataset.

    Returns:
    - df: pandas DataFrame, with new engineered cryptographic and attack-related features.
    """
    # Cryptographic feature engineering
    df['qr_key_size'] = np.random.choice([3328, 4096, 6528], size=len(df))
    df['qr_signature_length'] = np.random.randint(2000, 4000, size(len(df)))
    df['qr_encapsulation_time'] = np.random.uniform(0.001, 0.005, size=len(df))
    df['qr_decapsulation_time'] = np.random.uniform(0.001, 0.005, size=len(df))
    
    # Attack-related feature engineering
    df = engineer_attack_features(df)  # Add features like failed decapsulation attempts, access attempts, etc.
    
    return df

# Prepare sequences for LSTM input
def prepare_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Simulating transactions 
def simulate_swift_transactions(unsw, cicids, ieee_cis, paysim, n_transactions=10000):
    swift_transactions = []
    
    for _ in range(n_transactions):
        transaction = generate_synthetic_swift_data(n_samples=1).iloc[0]
        
        network_sample = unsw.sample(n=1).iloc[0] if np.random.random() < 0.5 else cicids.sample(n=1).iloc[0]
        transaction['src_ip'] = network_sample.get('srcip', network_sample.get('Source IP'))
        transaction['dst_ip'] = network_sample.get('dstip', network_sample.get('Destination IP'))
        transaction['protocol'] = network_sample.get('proto', network_sample.get('Protocol'))
        
        fraud_sample = ieee_cis.sample(n=1).iloc[0] if np.random.random() < 0.5 else paysim.sample(n=1).iloc[0]
        transaction['is_fraud'] = fraud_sample.get('isFraud', fraud_sample.get('isFlaggedFraud'))
        
        # Perform actual CRYSTALS-Kyber operations
        kyber_data = perform_kyber_operations()
        transaction.update(kyber_data)
        
        swift_transactions.append(transaction)
    
    return pd.DataFrame(swift_transactions)

# Synthetic SWIFT Data Generator 
def generate_synthetic_swift_data(n_samples=1000):
    faker = Faker()
    data = {
        'transaction_id': [faker.uuid4() for _ in range(n_samples)],
        'amount': [faker.random_number(digits=6) for _ in range(n_samples)],
        'sender': [faker.company() for _ in range(n_samples)],
        'receiver': [faker.company() for _ in range(n_samples)],
        'timestamp': [faker.date_time_this_year() for _ in range(n_samples)]
    }
    df = pd.DataFrame(data)
    return df

# Kyber Operations 
def perform_kyber_operations():
    try:
        public_key, secret_key = kyber512.keypair()
        message = np.random.bytes(32)
        ciphertext, shared_secret_enc = kyber512.encrypt(public_key, message)
        shared_secret_dec = kyber512.decrypt(secret_key, ciphertext)
    except Exception as e:
        print(f"Error during Kyber operations: {e}")
        return {}

    return {
        'public_key': public_key.hex(),
        'ciphertext': ciphertext.hex(),
        'shared_secret_enc': shared_secret_enc.hex(),
        'shared_secret_dec': shared_secret_dec.hex(),
        'key_size': len(public_key)
    }

# Differentially Private Model Training with Opacus
def train_privacy_preserving_model(train_data, target_data, batch_size=32, epochs=5):
    # Ensure no NaN values in the data
    if np.isnan(train_data).any() or np.isnan(target_data).any():
        raise ValueError("Input data contains NaNs. Please handle missing values before training.")
    
    # Define a more complex model
    model = nn.Sequential(
        nn.Linear(train_data.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    # Create DataLoader
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Initialize Opacus Privacy Engine
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch_size,
        sample_size=len(train_data),
        max_grad_norm=1.0
    )
    privacy_engine.attach(optimizer)

    # Training loop with differential privacy
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

    # Display privacy consumption
    epsilon, _ = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Training completed with epsilon: {epsilon}")

    # Return the trained model
    return model

# In the main function, preprocess and sort the data by timestamp
if __name__ == "__main__":
    # Load datasets 
    ieee_cis_data = load_ieee_cis_data('Data/IEEE-CIS-Fraud-Detection-master/train_transaction.csv')
    paysim_data = load_paysim_data('Data/PaySim-master/PS_20174392719_1491204439457_log.csv')

    # Preprocess datasets 
    ieee_cis_data = preprocess_data(ieee_cis_data)
    paysim_data = preprocess_data(paysim_data)

    # Engineer cryptographic and attack-related features
    ieee_cis_data = engineer_features(ieee_cis_data)
    paysim_data = engineer_features(paysim_data)

    # Combine data and sort by timestamp for LSTM
    combined_data = pd.concat([ieee_cis_data, paysim_data], axis=0).sort_values(by='timestamp')

    # Select features for LSTM
    X = combined_data[['amount', 'qr_key_size', 'qr_encapsulation_time', 'qr_decapsulation_time',
                       'failed_decapsulation_attempts', 'repeated_access_attempts', 'network_traffic_burst']].values
    y = combined_data['is_fraud'].values

    # Prepare sequences for LSTM
    sequence_length = 10  # Adjust based on the required sequence length
    X_seq, y_seq = prepare_sequences(X, sequence_length)

    # Continue with your model training process
