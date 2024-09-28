import time 
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay, precision_recall_curve
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

# New Metrics: Mean Average Precision (MAP) and Normalized Discounted Cumulative Gain (NDCG)
def mean_average_precision(y_true, y_pred):
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    average_precision = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    return average_precision

def dcg_score(y_true, y_pred, k=10):
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(1, len(y_true) + 1) + 1)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_pred, k=10):
    dcg = dcg_score(y_true, y_pred, k)
    ideal_dcg = dcg_score(y_true, y_true, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

# Define or Import adapt_lstm
def adapt_lstm(data, sequence_length, epochs=10, batch_size=32):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, data.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Data Preprocessing Functions
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
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    logger.info(f"Normalized data using {method} method")
    return df

# Evaluation Functions (Now include timing for predictions)
def evaluate_isolation_forest(model, X_test, y_true):
    try:
        start_time = time.time()  # Start timer for prediction
        y_pred = model.predict(X_test)
        end_time = time.time()  # End timer
        prediction_time = end_time - start_time
        logger.info(f"Prediction time for Isolation Forest: {prediction_time} seconds")  # Log time
        
        y_pred_binary = [1 if x == 1 else -1 for x in y_pred]
        
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        
        # Calculate MAP and NDCG for ranking evaluation
        y_pred_prob = model.decision_function(X_test)
        map_score = mean_average_precision(y_true, y_pred_prob)
        ndcg = ndcg_score(y_true, y_pred_prob)
        
        logger.info("Evaluated Isolation Forest model")
        return {"precision": precision, "recall": recall, "f1": f1, "map": map_score, "ndcg": ndcg}
    except NotFittedError:
        logger.error("Isolation Forest model is not fitted. Please train the model before evaluation.")
        return None

def evaluate_lstm(model, X_test, y_true):
    try:
        start_time = time.time()  # Start timer for prediction
        y_pred = model.predict(X_test)
        end_time = time.time()  # End timer
        prediction_time = end_time - start_time
        logger.info(f"Prediction time for LSTM: {prediction_time} seconds")  # Log time

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

# Visualization Functions
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

# Prepare sequences for LSTM
def prepare_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Ensemble Voting Mechanism
def ensemble_voting(isolation_forest_pred, lstm_pred, threshold=0.5):
    """
    Combines predictions from Isolation Forest and LSTM using simple voting.
    
    Parameters:
    - isolation_forest_pred: list, predictions from the Isolation Forest model.
    - lstm_pred: list, predictions from the LSTM model.
    - threshold: float, the decision threshold for LSTM predictions.
    
    Returns:
    - combined_pred: list, combined predictions from both models.
    """
    combined_pred = []
    for iso_pred, lstm_p in zip(isolation_forest_pred, lstm_pred):
        vote = (iso_pred + (1 if lstm_p > threshold else 0)) / 2
        combined_pred.append(1 if vote >= 0.5 else -1)
    return np.array(combined_pred)

# Ensemble Model Evaluation
def evaluate_ensemble(X_test, y_true, isolation_forest_model, lstm_model, sequence_length, threshold=0.5):
    """
    Evaluates the performance of the ensemble model using custom metrics.
    
    Parameters:
    - X_test: numpy array, test data.
    - y_true: numpy array, true labels.
    - isolation_forest_model: trained Isolation Forest model.
    - lstm_model: trained LSTM model.
    - sequence_length: int, the length of sequences for LSTM.
    - threshold: float, the decision threshold for LSTM predictions.
    
    Returns:
    - results: dict, contains precision, recall, and F1-score.
    """
    start_time = time.time()  # Start timer for predictions
    
    # Predict using Isolation Forest
    iso_pred = isolation_forest_model.predict(X_test)
    
    # Prepare sequences for LSTM and predict
    X_seq_test, _ = prepare_sequences(X_test, sequence_length)
    lstm_pred = lstm_model.predict(X_seq_test).flatten()
    
    # Combine predictions using voting mechanism
    combined_pred = ensemble_voting(iso_pred, lstm_pred, threshold)
    
    precision = precision_score(y_true, combined_pred)
    recall = recall_score(y_true, combined_pred)
    f1 = f1_score(y_true, combined_pred)
    
    end_time = time.time()  # End timer for predictions
    logger.info(f"Ensemble prediction time: {end_time - start_time} seconds")  # Log prediction time

    return {"precision": precision, "recall": recall, "f1": f1}

# ROC-AUC Evaluation
def evaluate_roc_auc(model, X_test, y_true):
    """
    Evaluates the ROC-AUC score for models using predict_proba.
    
    Parameters:
    - model: trained classifier model that supports predict_proba.
    - X_test: numpy array, test data.
    - y_true: numpy array, true labels.
    
    Returns:
    - roc_auc: float, ROC-AUC score.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # For classifiers with predict_proba
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    return roc_auc
# k-fold Cross-validation
def k_fold_cross_validation(model_func, X, y, k=5, sequence_length=None):
    """
    Perform k-fold cross-validation on the given model.

    Parameters:
    - model_func: callable, function that returns a new instance of the model to be trained.
    - X: numpy array, the input data.
    - y: numpy array, the target labels.
    - k: int, the number of folds for cross-validation.
    - sequence_length: int, sequence length for LSTM models (optional).

    Returns:
    - avg_metrics: dict, average precision, recall, f1, map, and ndcg metrics across all folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics = {"precision": [], "recall": [], "f1": [], "map": [], "ndcg": []}
    
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

# Bootstrap Sampling
def bootstrap_sampling(model_func, X, y, n_iterations=100, sequence_length=None):
    """
    Perform bootstrap sampling for model evaluation.

    Parameters:
    - model_func: callable, function that returns a new instance of the model to be trained.
    - X: numpy array, the input data.
    - y: numpy array, the target labels.
    - n_iterations: int, the number of bootstrap iterations.
    - sequence_length: int, sequence length for LSTM models (optional).

    Returns:
    - avg_metrics: dict, average precision, recall, f1, map, and ndcg metrics across all iterations.
    """
    metrics = {"precision": [], "recall": [], "f1": [], "map": [], "ndcg": []}
    
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
    """
    Evaluate cryptographic performance metrics such as key sizes, encapsulation times, and decapsulation times.
    """
    key_sizes = analyze_key_sizes()
    encapsulation_times = analyze_encapsulation_times()
    decapsulation_times = analyze_decapsulation_times()

    logger.info(f"Average Key Size: {np.mean(key_sizes)} bytes")
    logger.info(f"Average Encapsulation Time: {np.mean(encapsulation_times)} seconds")
    logger.info(f"Average Decapsulation Time: {np.mean(decapsulation_times)} seconds")

# Synthetic SWIFT Data Generator and Kyber Operations
def simulate_swift_transactions(unsw, cicids, ieee_cis, paysim, n_transactions=10000):
    """
    Simulate SWIFT transactions by combining multiple datasets and performing CRYSTALS-Kyber operations.

    Parameters:
    - unsw: pandas DataFrame, the UNSW dataset.
    - cicids: pandas DataFrame, the CICIDS dataset.
    - ieee_cis: pandas DataFrame, the IEEE-CIS dataset.
    - paysim: pandas DataFrame, the PaySim dataset.
    - n_transactions: int, number of transactions to simulate.

    Returns:
    - swift_transactions: pandas DataFrame, the generated SWIFT-like transactions.
    """
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
    """
    Generate synthetic SWIFT-like data.

    Parameters:
    - n_samples: int, the number of samples to generate.

    Returns:
    - pandas DataFrame, the generated synthetic transactions.
    """
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
    """
    Perform CRYSTALS-Kyber operations, including key generation, encapsulation, and decapsulation.

    Returns:
    - dict: containing public key, ciphertext, and shared secrets.
    """
    public_key, secret_key = kyber512.keypair()
    
    # Generate random message for encryption
    message = np.random.bytes(32)
    ciphertext, shared_secret_enc = kyber512.encrypt(public_key, message)
    
    # Decrypt to get the shared secret
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
    """
    Train a model with differential privacy using Opacus.

    Parameters:
    - train_data: numpy array, the training input data.
    - target_data: numpy array, the target labels.
    - batch_size: int, batch size for training.
    - epochs: int, number of epochs for training.

    Returns:
    - model: the trained PyTorch model.
    """
    model = nn.Sequential(
        nn.Linear(train_data.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Opacus Privacy Engine for differential privacy
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
    """
    Check for bias in the dataset using AIF360.

    Parameters:
    - data: pandas DataFrame, the dataset containing a binary label and protected attribute.

    Returns:
    - fair_dataset: BinaryLabelDataset, the reweighed dataset to correct bias.
    """
    dataset = BinaryLabelDataset(df=data, label_names=['label'], protected_attribute_names=['gender'])
    rw = Reweighing(unprivileged_groups=[{'gender': 0}], privileged_groups=[{'gender': 1}])
    fair_dataset = rw.fit_transform(dataset)
    logger.info("Performed fairness and bias check")
    return fair_dataset

# New function for model serialization
def save_model(model, filename):
    """
    Save a model to a file using joblib.

    Parameters:
    - model: the model to be saved.
    - filename: str, the path where the model should be saved.
    """
    try:
        joblib.dump(model, filename)
        logger.info(f"Model successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving model to {filename}: {str(e)}")

# New function for model deserialization
def load_model(filename):
    """
    Load a model from a file using joblib.

    Parameters:
    - filename: str, the path from where the model should be loaded.

    Returns:
    - model: the loaded model, or None if an error occurred.
    """
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

