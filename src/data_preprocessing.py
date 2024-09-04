import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pqcrypto.kem import kyber512
from faker import Faker
import torch
import syft
from syft.frameworks.torch import dp

# Load datasets
def load_ieee_cis_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    return df

def load_paysim_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    return df

def load_unsw_nb15_data(filepath):
    try:
        df = pd.read_parquet(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    return df

def load_cicids2017_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    return df

# Preprocessing functions
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

def engineer_features(df):
    df['qr_key_size'] = np.random.choice([3328, 4096, 6528], size=len(df))
    df['qr_signature_length'] = np.random.randint(2000, 4000, size=len(df))
    df['qr_encapsulation_time'] = np.random.uniform(0.001, 0.005, size=len(df))
    df['qr_decapsulation_time'] = np.random.uniform(0.001, 0.005, size=len(df))
    return df

def create_time_series(df, sequence_length):
    sequences = []
    for i in range(len(df) - sequence_length + 1):
        sequences.append(df.iloc[i:i+sequence_length].values)
    return np.array(sequences)

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

def preprocess_pipeline(unsw, cicids, ieee_cis, paysim):
    datasets = [unsw, cicids, ieee_cis, paysim]
    cleaned_datasets = [clean_data(df) for df in datasets]
    featured_datasets = [engineer_features(df) for df in cleaned_datasets]
    normalized_datasets = [normalize_data(df, method='minmax') for df in featured_datasets]
    
    swift_data = simulate_swift_transactions(*normalized_datasets)
    
    sequence_length = 10
    time_series_data = create_time_series(swift_data, sequence_length)
    
    return swift_data, time_series_data

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

def add_kyber_elements(df):
    public_keys = []
    ciphertexts = []
    for _ in range(len(df)):
        public_key, secret_key = kyber512.keypair()
        ciphertext, shared_secret = kyber512.encrypt(public_key)
        public_keys.append(public_key.hex())
        ciphertexts.append(ciphertext.hex())
    df['public_key'] = public_keys
    df['ciphertext'] = ciphertexts
    return df

def perform_kyber_operations():
    public_key, secret_key = kyber512.keypair()
    
    message = np.random.bytes(32)
    ciphertext, shared_secret_enc = kyber512.encrypt(public_key, message)
    
    shared_secret_dec = kyber512.decrypt(secret_key, ciphertext)
    
    return {
        'public_key': public_key.hex(),
        'ciphertext': ciphertext.hex(),
        'shared_secret_enc': shared_secret_enc.hex(),
        'shared_secret_dec': shared_secret_dec.hex(),
        'key_size': len(public_key)
    }

# New function for adding noise (differential privacy)
def add_noise(data):
    noise = torch.randn_like(data) * 0.1
    return data + noise

if __name__ == "__main__":
    # Load datasets from their respective paths
    ieee_cis_data = load_ieee_cis_data('Data/IEEE-CIS-Fraud-Detection-master/train_transaction.csv')
    paysim_data = load_paysim_data('Data/PaySim-master/PS_20174392719_1491204439457_log.csv')
    unsw_nb15_data = load_unsw_nb15_data('Data/UNSW-NB15/UNSW_NB15_training-set.parquet')
    cicids2017_data = load_cicids2017_data('Data/CIC-IDS2017/your_cicids2017_file.csv')
    
    # Preprocess each dataset
    ieee_cis_data = preprocess_data(ieee_cis_data)
    paysim_data = preprocess_data(paysim_data)
    unsw_nb15_data = preprocess_data(unsw_nb15_data)
    cicids2017_data = preprocess_data(cicids2017_data)
    
    # Generate synthetic SWIFT data and add CRYSTALS-Kyber elements
    swift_data = generate_synthetic_swift_data()
    swift_data = add_kyber_elements(swift_data)
    
    print("Data preprocessing complete.")
    
    # Example of displaying processed data
    print(swift_data.head())

    # New code: Apply differential privacy
    swift_data_tensor = torch.tensor(swift_data.select_dtypes(include=[np.number]).values, dtype=torch.float32)
    swift_data_with_noise = add_noise(swift_data_tensor)
    swift_data.loc[:, swift_data.select_dtypes(include=[np.number]).columns] = swift_data_with_noise.numpy()
    
    print("Differential privacy applied.")
    print(swift_data.head())