import time
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
from sklearn.utils import resample
from src.crypto_analysis import analyze_key_sizes, analyze_encapsulation_times, analyze_decapsulation_times
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pqcrypto.kem import kyber512
from faker import Faker
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import logging
from multiprocessing import Pool
from functools import partial
import yaml
import joblib
from sklearn.exceptions import NotFittedError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Utility Function for Profiling
def profile_code(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    logger.info(f"Execution time for {func.__name__}: {end_time - start_time} seconds")
    return result

# Data Loading Functions
def load_ieee_cis_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded IEEE-CIS data from {filepath}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: {filepath} not found.")
        return None

def load_paysim_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded PaySim data from {filepath}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: {filepath} not found.")
        return None

# Data Preprocessing Functions
def preprocess_data(df):
    if df is None:
        logger.warning("Received None dataframe for preprocessing. Returning None.")
        return None
    df = clean_data(df)
    df = normalize_data(df)
    return df

def clean_data(df):
    original_shape = df.shape
    df = df.drop_duplicates()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())
    logger.info(f"Cleaned data. Rows removed: {original_shape[0] - df.shape[0]}")
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
    logger.info(f"Normalized data using {method} method")
    return df

def engineer_features(df):
    df['qr_key_size'] = np.random.choice([3328, 4096, 6528], size=len(df))
    df['qr_signature_length'] = np.random.randint(2000, 4000, size=len(df))
    df['qr_encapsulation_time'] = np.random.uniform(0.001, 0.005, size=len(df))
    df['qr_decapsulation_time'] = np.random.uniform(0.001, 0.005, size=len(df))
    logger.info("Engineered new features for quantum resistance analysis")
    return df

# Model Adaptation Functions
def adapt_isolation_forest(data, contamination=0.1):
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(data)
    logger.info("Adapted Isolation Forest model")
    return clf

def adapt_lstm(data, sequence_length, epochs=10, batch_size=32):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, data.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    logger.info("Adapted LSTM model")
    return model

def prepare_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Evaluation Functions
def evaluate_isolation_forest(model, X_test, y_true):
    try:
        y_pred = model.predict(X_test)
        y_pred_binary = [1 if x == 1 else -1 for x in y_pred]
        
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        
        logger.info("Evaluated Isolation Forest model")
        return {"precision": precision, "recall": recall, "f1": f1}
    except NotFittedError:
        logger.error("Isolation Forest model is not fitted. Please train the model before evaluation.")
        return None

def evaluate_lstm(model, X_test, y_true):
    try:
        y_pred = model.predict(X_test)
        mse = np.mean((y_true - y_pred)**2)
        
        threshold = np.mean(mse) + 2 * np.std(mse)
        y_pred_binary = [1 if x > threshold else -1 for x in mse]
        
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        
        logger.info("Evaluated LSTM model")
        return {"mse": mse, "precision": precision, "recall": recall, "f1": f1}
    except Exception as e:
        logger.error(f"Error evaluating LSTM model: {str(e)}")
        return None

# New visualization functions for ROC Curve and Confusion Matrix
def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def k_fold_cross_validation(model_func, X, y, k=5, sequence_length=None):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics = {"precision": [], "recall": [], "f1": []}
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if model_func == adapt_lstm:
            X_train, y_train = prepare_sequences(X_train, sequence_length)
            X_test, y_test = prepare_sequences(X_test, sequence_length)
            model = model_func(X_train, sequence_length)
        else:
            model = model_func(X_train)
        
        model.fit(X_train, y_train)
        results = evaluate_lstm(model, X_test, y_test) if model_func == adapt_lstm else evaluate_isolation_forest(model, X_test, y_test)
        
        for key in metrics.keys():
            metrics[key].append(results[key])
    
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    logger.info(f"Completed {k}-fold cross-validation")
    return avg_metrics

def bootstrap_sampling(model_func, X, y, n_iterations=100, sequence_length=None):
    metrics = {"precision": [], "recall": [], "f1": []}
    
    for i in range(n_iterations):
        X_resample, y_resample = resample(X, y, n_samples=len(X), random_state=i)
        X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size=0.2, random_state=i)
        
        if model_func == adapt_lstm:
            X_train, y_train = prepare_sequences(X_train, sequence_length)
            X_test, y_test = prepare_sequences(X_test, sequence_length)
            model = model_func(X_train, sequence_length)
        else:
            model = model_func(X_train)
        
        model.fit(X_train, y_train)
        results = evaluate_lstm(model, X_test, y_test) if model_func == adapt_lstm else evaluate_isolation_forest(model, X_test, y_test)
        
        for key in metrics.keys():
            metrics[key].append(results[key])
    
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    logger.info(f"Completed bootstrap sampling with {n_iterations} iterations")
    return avg_metrics

# Cryptographic Performance Evaluation
def evaluate_crypto_performance():
    key_sizes = analyze_key_sizes()
    encapsulation_times = analyze_encapsulation_times()
    decapsulation_times = analyze_decapsulation_times()

    logger.info(f"Average Key Size: {np.mean(key_sizes)} bytes")
    logger.info(f"Average Encapsulation Time: {np.mean(encapsulation_times)} seconds")
    logger.info(f"Average Decapsulation Time: {np.mean(decapsulation_times)} seconds")

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
    
    logger.info(f"Simulated {n_transactions} SWIFT transactions")
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
    logger.info(f"Generated {n_samples} synthetic SWIFT transactions")
    return df

# Kyber Operations 
def perform_kyber_operations():
    public_key, secret_key = kyber512.keypair()
    
    message = np.random.bytes(32)
    ciphertext, shared_secret_enc = kyber512.encrypt(public_key, message)
    
    shared_secret_dec = kyber512.decrypt(secret_key, ciphertext)
    
    logger.info("Performed CRYSTALS-Kyber operations")
    return {
        'public_key': public_key.hex(),
        'ciphertext': ciphertext.hex(),
        'shared_secret_enc': shared_secret_enc.hex(),
        'shared_secret_dec': shared_secret_dec.hex(),
        'key_size': len(public_key)
    }

# Differentially Private Model Training with Opacus
def train_privacy_preserving_model(train_data, target_data, batch_size=32, epochs=5):
    model = nn.Sequential(
        nn.Linear(train_data.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch_size,
        sample_size=len(train_data),
        max_grad_norm=1.0
    )
    privacy_engine.attach(optimizer)

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

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

    logger.info("Completed privacy-preserving model training")
    return model

# Fairness and Bias Check using AIF360
def check_for_bias(data):
    dataset = BinaryLabelDataset(df=data, label_names=['label'], protected_attribute_names=['gender'])
    rw = Reweighing(unprivileged_groups=[{'gender': 0}], privileged_groups=[{'gender': 1}])
    fair_dataset = rw.fit_transform(dataset)
    logger.info("Performed fairness and bias check")
    return fair_dataset


# New function for model serialization
def save_model(model, filename):
    try:
        joblib.dump(model, filename)
        logger.info(f"Model successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving model to {filename}: {str(e)}")

# New function for model deserialization
def load_model(filename):
    try:
        model = joblib.load(filename)
        logger.info(f"Model successfully loaded from {filename}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {filename}")
        return None
    except Exception as e:
        logger.error(f"Error loading model from {filename}: {str(e)}")
        return None

