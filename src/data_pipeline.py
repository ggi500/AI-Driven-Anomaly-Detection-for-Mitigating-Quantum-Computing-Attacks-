import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from faker import Faker
from pqcrypto.kem.kyber512 import generate_keypair, encrypt, decrypt
import time
import os

fake = Faker()

# Import your load_datasets function
from data_acquisition import load_datasets

def clean_data(df):
    df = df.drop_duplicates()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())
    return df

def engineer_features(df):
    df['qr_key_size'] = np.random.choice([3328, 4096, 6528], size=len(df))
    df['qr_signature_length'] = np.random.randint(2000, 4000, size=len(df))
    df['qr_encapsulation_time'] = np.random.uniform(0.001, 0.005, size=len(df))
    df['qr_decapsulation_time'] = np.random.uniform(0.001, 0.005, size=len(df))
    return df

def normalize_data(df, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalisation method. Choose 'minmax' or 'zscore'.")
    
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def create_time_series(df, sequence_length):
    sequences = []
    for i in range(len(df) - sequence_length + 1):
        sequences.append(df.iloc[i:i+sequence_length].values)
    return np.array(sequences)

def generate_swift_transaction():
    return {
        'transaction_id': fake.uuid4(),
        'sender_bic': fake.swift8(),
        'receiver_bic': fake.swift8(),
        'amount': round(np.random.uniform(100, 1000000), 2),
        'currency': np.random.choice(['USD', 'EUR', 'GBP', 'JPY', 'CHF']),
        'timestamp': fake.date_time_this_year(),
    }

def perform_kyber_operations():
    # Generate a keypair
    public_key, secret_key = generate_keypair()
    
    # Encrypt a random message
    message = np.random.bytes(32)
    start_time = time.time()
    ciphertext, shared_secret_enc = encrypt(public_key, message)
    encryption_time = time.time() - start_time
    
    # Decrypt the message
    start_time = time.time()
    shared_secret_dec = decrypt(secret_key, ciphertext)
    decryption_time = time.time() - start_time
    
    return {
        'public_key': public_key.hex(),
        'ciphertext': ciphertext.hex(),
        'encryption_time': encryption_time,
        'decryption_time': decryption_time,
        'key_size': len(public_key)
    }

def simulate_swift_transactions(unsw, cicids, ieee_cis, paysim, n_transactions=10000):
    swift_transactions = []
    
    for _ in range(n_transactions):
        transaction = generate_swift_transaction()
        
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

def generate_and_save_data():
    print("Generating data...")
    
    # Ensure Data directory exists
    data_dir = os.path.join(os.getcwd(), 'Data')
    os.makedirs(data_dir, exist_ok=True)

    try:
        # Load and preprocess multiple datasets
        print("Loading and preprocessing data...")
        unsw, cicids, ieee_cis, paysim = load_datasets()
        swift_data, time_series_data = preprocess_pipeline(unsw, cicids, ieee_cis, paysim)
        
        print("Data preprocessing complete.")
        print("Shape of preprocessed SWIFT-like data:", swift_data.shape)
        print("Shape of time series data:", time_series_data.shape)
        
        # Save the processed data to the Data directory
        swift_data_path = os.path.join(data_dir, 'processed_swift_data.csv')
        time_series_data_path = os.path.join(data_dir, 'time_series_data.npy')
        
        swift_data.to_csv(swift_data_path, index=False)
        np.save(time_series_data_path, time_series_data)
        
        print(f"Processed data saved to {swift_data_path}")
        print(f"Time series data saved to {time_series_data_path}")
        
        return swift_data, time_series_data
    
    except Exception as e:
        print(f"An error occurred during data generation: {str(e)}")
        return None, None

if __name__ == "__main__":
    generate_and_save_data()