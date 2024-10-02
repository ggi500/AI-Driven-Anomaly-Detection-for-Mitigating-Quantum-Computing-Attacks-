import time  
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.utils import resample
from src.crypto_analysis import analyze_key_sizes, analyze_encapsulation_times, analyze_decapsulation_times
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pqcrypto.kem import kyber512
from faker import Fake
from scipy.stats import ks_2samp
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
import os
import tempfile

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
    combined_pred = []
    for iso_pred, lstm_p in zip(isolation_forest_pred, lstm_pred):
        vote = (iso_pred + (1 if lstm_p > threshold else 0)) / 2
        combined_pred.append(1 if vote >= 0.5 else -1)
    return np.array(combined_pred)

# Ensemble Model Evaluation
def evaluate_ensemble(X_test, y_true, isolation_forest_model, lstm_model, sequence_length, threshold=0.5):
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
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # For classifiers with predict_proba
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    return roc_auc

# k-fold Cross-validation
def k_fold_cross_validation(model_func, X, y, k=5, sequence_length=None):
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
    key_sizes = analyze_key_sizes()
    encapsulation_times = analyze_encapsulation_times()
    decapsulation_times = analyze_decapsulation_times()

    logger.info(f"Average Key Size: {np.mean(key_sizes)} bytes")
    logger.info(f"Average Encapsulation Time: {np.mean(encapsulation_times)} seconds")
    logger.info(f"Average Decapsulation Time: {np.mean(decapsulation_times)} seconds")

# Synthetic SWIFT Data Generator and Kyber Operations
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
    
    # Generate random message for encryption
    message = np.random
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

# New functions for checkpointing and rollback

def create_isolation_forest_checkpoint(model, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"isolation_forest_checkpoint_{int(time.time())}.joblib")
    joblib.dump(model, checkpoint_path)
    logger.info(f"Created Isolation Forest checkpoint: {checkpoint_path}")
    return checkpoint_path

def adapt_lstm(data, sequence_length, epochs=10, batch_size=32, checkpoint_dir='checkpoints/lstm'):
    def train_lstm():
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, data.shape[2])),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Create a temporary validation set
        val_split = 0.2
        val_samples = int(len(data) * val_split)
        X_train, X_val = data[:-val_samples], data[-val_samples:]

        # Training without the ModelCheckpoint callback
        model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, X_val))

        # Manually saving the model's weights using PyTorch after training
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "lstm_model_weights.pth")
        torch.save(model.get_weights(), checkpoint_path)  # Save the model's weights with PyTorch

        return model, checkpoint_path

    model, checkpoint_path = profile_code(train_lstm)

    # Load the weights back into the model using PyTorch's torch.load
    model.set_weights(torch.load(checkpoint_path))

    return model, checkpoint_path

def rollback_isolation_forest(checkpoint_path):
    if os.path.exists(checkpoint_path):
        model = joblib.load(checkpoint_path)
        logger.info(f"Rolled back Isolation Forest model to: {checkpoint_path}")
        return model
    else:
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None

def rollback_lstm(checkpoint_path):
    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)
        logger.info(f"Rolled back LSTM model to: {checkpoint_path}")
        return model
    else:
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None

# Modified adapt_isolation_forest function with checkpointing
def adapt_isolation_forest(data, contamination=0.1, checkpoint_dir='checkpoints/isolation_forest'):
    def train_isolation_forest():
        clf = IsolationForest(contamination=contamination, random_state=42)
        clf.fit(data)
        return clf

    clf = profile_code(train_isolation_forest)
    
    # Create a checkpoint
    checkpoint_path = create_isolation_forest_checkpoint(clf, checkpoint_dir)
    
    return clf, checkpoint_path
# Function to detect data drift
def detect_drift(old_data, new_data, p_value_threshold=0.05):
    """
    Detects drift between old_data and new_data using the Kolmogorov-Smirnov test.

    Parameters:
    old_data (numpy.ndarray): The data that was used for training.
    new_data (numpy.ndarray): The new data to compare for drift.
    p_value_threshold (float): The threshold for detecting drift. If the p-value of the KS test is 
                               less than this threshold, drift is detected.

    Returns:
    bool: True if drift is detected, False otherwise.
    """
    drift_detected = False
    for i in range(old_data.shape[1]):  # Assuming old_data and new_data have the same number of features
        stat, p_value = ks_2samp(old_data[:, i], new_data[:, i])
        if p_value < p_value_threshold:
            drift_detected = True
            break
    return drift_detected

# Function for continuous retraining with drift detection
def continuous_retraining_with_drift_detection(model, X_new, y_new, old_data, model_type, retrain_frequency=100, p_value_threshold=0.05):
    if len(X_new) % retrain_frequency == 0:
        if detect_drift(old_data, X_new, p_value_threshold):
            logger.info("Data drift detected. Retraining the model.")
            # Proceed with checkpointing and retraining logic

def adapt_lstm(data, sequence_length, epochs=10, batch_size=32, checkpoint_dir='checkpoints/lstm'):
    def train_lstm():
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, data.shape[2])),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Create a temporary validation set
        val_split = 0.2
        val_samples = int(len(data) * val_split)
        X_train, X_val = data[:-val_samples], data[-val_samples:]
        
        # Training without TensorFlow's ModelCheckpoint
        model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, X_val))
        
        # Manually saving the model weights using PyTorch after training
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "lstm_model_weights.pth")
        torch.save(model.get_weights(), checkpoint_path)  # Save Keras weights using PyTorch
        
        return model, checkpoint_path

    # Train the model and save the checkpoint
    model, checkpoint_path = profile_code(train_lstm)
    
    # Load the weights back into the model using PyTorch's torch.load
    model.set_weights(torch.load(checkpoint_path))
    
    return model, checkpoint_path
# Modified continuous_retraining_with_drift_detection function with checkpointing and rollback
def continuous_retraining_with_drift_detection(model, X_new, y_new, old_data, model_type, retrain_frequency=100, p_value_threshold=0.05):
    if len(X_new) % retrain_frequency == 0:
        if detect_drift(old_data, X_new, p_value_threshold):
            logger.info("Data drift detected. Retraining the model.")
            
            # Create a checkpoint before retraining
            if model_type == 'isolation_forest':
                checkpoint_path = create_isolation_forest_checkpoint(model, 'checkpoints/isolation_forest')
            elif model_type == 'lstm':
                # Manual checkpointing for LSTM using PyTorch's torch.save
                checkpoint_dir = 'checkpoints/lstm'
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "lstm_model_weights.pth")
                torch.save(model.get_weights(), checkpoint_path)  # Save LSTM model weights using PyTorch
                
            try:
                if model_type == 'isolation_forest':
                    model, _ = adapt_isolation_forest(X_new)
                elif model_type == 'lstm':
                    sequence_length = X_new.shape[1]  # Assume X_new is already sequenced
                    model, _ = adapt_lstm(X_new, sequence_length)

                logger.info("Model retraining complete due to detected drift.")
            except Exception as e:
                logger.error(f"Error during retraining: {str(e)}. Rolling back to previous checkpoint.")
                if model_type == 'isolation_forest':
                    model = rollback_isolation_forest(checkpoint_path)
                elif model_type == 'lstm':
                    model = rollback_lstm(checkpoint_path)
        else:
            logger.info("No significant drift detected. Skipping retraining.")
    return model
# Main function for training and evaluation
if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = np.random.rand(1000, 10)  # 1000 samples, 10 features
    sequence_length = 5
    
    # Train Isolation Forest model with checkpointing
    isolation_forest_model, if_checkpoint_path = adapt_isolation_forest(data)
    
    # Prepare data for LSTM
    X_seq, y_seq = prepare_sequences(data, sequence_length)
    
    # Train LSTM model with checkpointing
    lstm_model, lstm_checkpoint_path = adapt_lstm(X_seq, sequence_length)

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(data, y_seq, test_size=0.2, random_state=42)

    # Ensemble evaluation
    ensemble_results = profile_code(evaluate_ensemble, X_test, y_test, isolation_forest_model, lstm_model, sequence_length)
    logger.info("Ensemble model evaluation: %s", ensemble_results)

    # Evaluate ROC-AUC for the Isolation Forest model
    roc_auc = profile_code(evaluate_roc_auc, isolation_forest_model, X_test, y_test)
    logger.info(f"Isolation Forest ROC-AUC: {roc_auc}")
    
    # Evaluate cryptographic performance
    profile_code(evaluate_crypto_performance)

    # Example of continuous retraining with drift detection and potential rollback
    new_data = np.random.rand(200, 10)  # New data batch
    isolation_forest_model = continuous_retraining_with_drift_detection(
        isolation_forest_model, new_data, None, X_train, 'isolation_forest')
    
    # Example of manual rollback
    rolled_back_if_model = rollback_isolation_forest(if_checkpoint_path)
    rolled_back_lstm_model = rollback_lstm(lstm_checkpoint_path)

    logger.info("Model adaptation, evaluation, checkpointing, and rollback demonstration complete.")