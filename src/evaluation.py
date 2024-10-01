import time   
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,
                             confusion_matrix, roc_curve, ConfusionMatrixDisplay, precision_recall_curve)
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import NotFittedError
from src.crypto_analysis import analyze_key_sizes, analyze_encapsulation_times, analyze_decapsulation_times
from scipy.stats import ks_2samp  # For drift detection
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from faker import Faker
from pqcrypto.kem import kyber512
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import yaml
from multiprocessing import Pool
from functools import partial
from sklearn.metrics import mean_squared_error
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Utility Function for Profiling
def profile_code(func, *args, **kwargs):
    """
    Profile the execution time of a function.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    logger.info(f"Execution time for {func.__name__}: {end_time - start_time} seconds")
    return result

# Drift Detection Function
def detect_drift(old_data, new_data, p_value_threshold=0.05):
    """
    Detects drift by performing the Kolmogorov-Smirnov test.
    
    Parameters:
    - old_data: numpy array, the historical data.
    - new_data: numpy array, the new data to check for drift.
    - p_value_threshold: float, the p-value threshold for drift detection.
    
    Returns:
    - bool: True if drift is detected, False otherwise.
    """
    drift_detected = False
    for col in range(old_data.shape[1]):
        stat, p_value = ks_2samp(old_data[:, col], new_data[:, col])
        if p_value < p_value_threshold:
            drift_detected = True
            logger.info(f"Drift detected in feature {col} with p-value {p_value}")
            break
    return drift_detected

# New Metrics: Mean Average Precision (MAP) and Normalized Discounted Cumulative Gain (NDCG)
def mean_average_precision(y_true, y_scores):
    """
    Calculate the mean average precision (MAP) for binary classification.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    average_precision = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    return average_precision

def dcg_score(y_true, y_scores, k=10):
    """
    Calculate the Discounted Cumulative Gain (DCG) at rank k.
    """
    order = np.argsort(y_scores)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(2, len(y_true) + 2))
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_scores, k=10):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at rank k.
    """
    dcg = dcg_score(y_true, y_scores, k)
    ideal_dcg = dcg_score(y_true, y_true, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

# Data Preprocessing Functions
def clean_data(df):
    """
    Clean the data by handling missing values and duplicates.
    """
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
    """
    Normalize numerical columns in the dataframe.
    """
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    logger.info(f"Normalized data using {method} method")
    return df

# Define or Import adapt_lstm
def adapt_lstm(sequence_length, input_dim):
    """
    Build and return an LSTM model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, input_dim)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare sequences for LSTM
def prepare_sequences(data, sequence_length):
    """
    Prepare sequences of data for LSTM input.
    """
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Evaluation Functions (with Drift Detection)
def evaluate_isolation_forest(model, X_test, y_true, X_train=None, p_value_threshold=0.05):
    """
    Evaluate Isolation Forest model, including drift detection.
    """
    try:
        # Drift detection between training data and test data
        if X_train is not None and detect_drift(X_train, X_test, p_value_threshold):
            logger.warning("Data drift detected between training and test data!")
        
        start_time = time.time()  # Start timer for prediction
        y_pred = model.predict(X_test)
        end_time = time.time()  # End timer
        prediction_time = end_time - start_time
        logger.info(f"Prediction time for Isolation Forest: {prediction_time} seconds")
        
        y_pred_binary = [1 if x == 1 else -1 for x in y_pred]
        
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        
        # Calculate MAP and NDCG for ranking evaluation
        y_pred_scores = model.decision_function(X_test)
        map_score = mean_average_precision(y_true, y_pred_scores)
        ndcg = ndcg_score(y_true, y_pred_scores)
        
        logger.info("Evaluated Isolation Forest model")
        return {"precision": precision, "recall": recall, "f1": f1, "map": map_score, "ndcg": ndcg}
    except NotFittedError:
        logger.error("Isolation Forest model is not fitted. Please train the model before evaluation.")
        return None

def evaluate_lstm(model, X_test, y_true, X_train=None, p_value_threshold=0.05):
    """
    Evaluate LSTM model, including drift detection.
    """
    try:
        # Drift detection between training data and test data
        if X_train is not None and detect_drift(X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1), p_value_threshold):
            logger.warning("Data drift detected between training and test data!")
        
        start_time = time.time()  # Start timer for prediction
        y_pred = model.predict(X_test)
        end_time = time.time()  # End timer
        prediction_time = end_time - start_time
        logger.info(f"Prediction time for LSTM: {prediction_time} seconds")
        
        mse = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
        
        threshold = np.mean(mse) + 2 * np.std(mse)
        y_pred_binary = [1 if x > threshold else -1 for x in mse]
        
        precision = precision_score(y_true.reshape(-1), y_pred_binary)
        recall = recall_score(y_true.reshape(-1), y_pred_binary)
        f1 = f1_score(y_true.reshape(-1), y_pred_binary)
        
        logger.info("Evaluated LSTM model")
        return {"mse": mse, "precision": precision, "recall": recall, "f1": f1}
    except Exception as e:
        logger.error(f"Error evaluating LSTM model: {str(e)}")
        return None

# Visualization Functions
def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

# Ensemble Voting Mechanism
def ensemble_voting(isolation_forest_pred, lstm_pred, threshold=0.5):
    """
    Combines predictions from Isolation Forest and LSTM using simple voting.
    """
    combined_pred = []
    for iso_pred, lstm_p in zip(isolation_forest_pred, lstm_pred):
        vote = (iso_pred + (1 if lstm_p > threshold else -1)) / 2
        combined_pred.append(1 if vote >= 0 else -1)
    return np.array(combined_pred)

# Ensemble Model Evaluation (with Drift Detection)
def evaluate_ensemble(X_test, y_true, isolation_forest_model, lstm_model, sequence_length, X_train=None, threshold=0.5, p_value_threshold=0.05):
    """
    Evaluates the performance of the ensemble model with drift detection.
    """
    # Drift detection between training data and test data
    if X_train is not None and detect_drift(X_train, X_test, p_value_threshold):
        logger.warning("Data drift detected between training and test data!")
    
    start_time = time.time()  # Start timer for predictions
    
    # Predict using Isolation Forest
    iso_pred = isolation_forest_model.predict(X_test)
    
    # Prepare sequences for LSTM and predict
    X_seq_test, _ = prepare_sequences(X_test, sequence_length)
    lstm_pred = lstm_model.predict(X_seq_test).flatten()
    
    # Combine predictions using voting mechanism
    combined_pred = ensemble_voting(iso_pred, lstm_pred, threshold)
    
    precision = precision_score(y_true[sequence_length:], combined_pred)
    recall = recall_score(y_true[sequence_length:], combined_pred)
    f1 = f1_score(y_true[sequence_length:], combined_pred)
    
    end_time = time.time()  # End timer for predictions
    logger.info(f"Ensemble prediction time: {end_time - start_time} seconds")
    
    return {"precision": precision, "recall": recall, "f1": f1}

# ROC-AUC Evaluation
def evaluate_roc_auc(model, X_test, y_true, X_train=None, p_value_threshold=0.05):
    """
    Evaluates the ROC-AUC score with drift detection.
    """
    # Drift detection between training data and test data
    if X_train is not None and detect_drift(X_train, X_test, p_value_threshold):
        logger.warning("Data drift detected between training and test data!")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # For classifiers with predict_proba
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    return roc_auc

# k-fold Cross-validation
def k_fold_cross_validation(model_func, X, y, k=5, sequence_length=None):
    """
    Perform k-fold cross-validation on the given model.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics = {"precision": [], "recall": [], "f1": [], "map": [], "ndcg": []}
    
    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        if model_func == adapt_lstm:
            X_train_seq, y_train_seq = prepare_sequences(X_train_fold, sequence_length)
            X_test_seq, y_test_seq = prepare_sequences(X_test_fold, sequence_length)
            model = model_func(sequence_length, X_train_seq.shape[2])
            model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32)
            results = evaluate_lstm(model, X_test_seq, y_test_seq)
        else:
            model = model_func()
            model.fit(X_train_fold, y_train_fold)
            results = evaluate_isolation_forest(model, X_test_fold, y_test_fold)
        
        for key in metrics.keys():
            metrics[key].append(results[key])
    
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    logger.info(f"Completed {k}-fold cross-validation")
    return avg_metrics

# Bootstrap Sampling
def bootstrap_sampling(model_func, X, y, n_iterations=100, sequence_length=None):
    """
    Perform bootstrap sampling for model evaluation.
    """
    metrics = {"precision": [], "recall": [], "f1": [], "map": [], "ndcg": []}
    
    for i in range(n_iterations):
        X_resample, y_resample = resample(X, y, n_samples=len(X), random_state=i)
        X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_resample, y_resample, test_size=0.2, random_state=i)
        
        if model_func == adapt_lstm:
            X_train_seq, y_train_seq = prepare_sequences(X_train_res, sequence_length)
            X_test_seq, y_test_seq = prepare_sequences(X_test_res, sequence_length)
            model = model_func(sequence_length, X_train_seq.shape[2])
            model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32)
            results = evaluate_lstm(model, X_test_seq, y_test_seq)
        else:
            model = model_func()
            model.fit(X_train_res, y_train_res)
            results = evaluate_isolation_forest(model, X_test_res, y_test_res)
        
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

# Synthetic SWIFT Data Generator
def generate_synthetic_swift_data(n_samples=1000):
    """
    Generate synthetic SWIFT-like data.
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
    """
    dataset = BinaryLabelDataset(df=data, label_names=['label'], protected_attribute_names=['gender'])
    rw = Reweighing(unprivileged_groups=[{'gender': 0}], privileged_groups=[{'gender': 1}])
    fair_dataset = rw.fit_transform(dataset)
    logger.info("Performed fairness and bias check")
    return fair_dataset

# Model Serialization Functions
def save_model(model, filename):
    """
    Save a model to a file using joblib.
    """
    try:
        joblib.dump(model, filename)
        logger.info(f"Model successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving model to {filename}: {str(e)}")

def load_model(filename):
    """
    Load a model from a file using joblib.
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

# Main Function for Evaluation
def main_evaluation():
    """
    Main function to perform evaluation of models.
    """
    # Load or generate data
    data = generate_synthetic_swift_data(n_samples=5000)
    data = clean_data(data)
    data = normalize_data(data)
    
    # Split features and target
    X = data.drop(columns=['transaction_id', 'timestamp', 'is_fraud']).values
    y = data['is_fraud'].values
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Isolation Forest
    isolation_forest_model = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest_model.fit(X_train)
    
    # Evaluate Isolation Forest
    isolation_forest_results = evaluate_isolation_forest(isolation_forest_model, X_test, y_test, X_train=X_train)
    logger.info(f"Isolation Forest Evaluation Results: {isolation_forest_results}")
    
    # Prepare sequences for LSTM
    sequence_length = 10
    X_seq_train, y_seq_train = prepare_sequences(X_train, sequence_length)
    X_seq_test, y_seq_test = prepare_sequences(X_test, sequence_length)
    
    # Train LSTM
    lstm_model = adapt_lstm(sequence_length, X_seq_train.shape[2])
    lstm_model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32)
    
    # Evaluate LSTM
    lstm_results = evaluate_lstm(lstm_model, X_seq_test, y_seq_test, X_train=X_seq_train)
    logger.info(f"LSTM Evaluation Results: {lstm_results}")
    
    # Evaluate Ensemble Model
    ensemble_results = evaluate_ensemble(X_test, y_test, isolation_forest_model, lstm_model, sequence_length, X_train=X_train)
    logger.info(f"Ensemble Model Evaluation Results: {ensemble_results}")
    
    # Evaluate ROC-AUC
    roc_auc = evaluate_roc_auc(isolation_forest_model, X_test, y_test, X_train=X_train)
    logger.info(f"Isolation Forest ROC-AUC: {roc_auc}")
    
    # Perform Cryptographic Performance Evaluation
    evaluate_crypto_performance()
    
    # Save models
    save_model(isolation_forest_model, 'models/isolation_forest_model.pkl')
    lstm_model.save('models/lstm_model.h5')
    
    # Load models
    loaded_if_model = load_model('models/isolation_forest_model.pkl')
    loaded_lstm_model = tf.keras.models.load_model('models/lstm_model.h5')

# Run the main evaluation function if this script is executed
if __name__ == "__main__":
    main_evaluation()
