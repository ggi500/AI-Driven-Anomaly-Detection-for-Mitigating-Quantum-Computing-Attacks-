import os  
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import pickle
import joblib
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Utility Function for Profiling
def profile_code(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    return result

# Model Saving Functions
def save_isolation_forest_model(model, version='v1'):
    """
    Saves the trained Isolation Forest model to the Models directory.
    """
    with open(f'Models/isolation_forest_model_{version}.pkl', 'wb') as file:
        pickle.dump(model, file)
    print(f"Isolation Forest model (version: {version}) saved.")

def save_lstm_model(model, version='final'):
    """
    Saves the trained LSTM model to the Models directory.
    """
    model.save(f'Models/lstm_model_{version}.h5')
    print(f"LSTM model (version: {version}) saved.")

# Function to save Isolation Forest
def train_and_save_isolation_forest(X_train, model_save_path):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    # Save model to `trained_models` directory
    joblib.dump(model, model_save_path)
    print(f"Isolation Forest model saved at {model_save_path}")

# Function to save LSTM model
def train_and_save_lstm(X_train, y_train, sequence_length, model_save_path):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(sequence_length, X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save model to `trained_models` directory
    model.save(model_save_path)
    print(f"LSTM model saved at {model_save_path}")

# Model Adaptation Functions
def adapt_isolation_forest(data, contamination=0.1):
    """
    Adapts and fine-tunes an Isolation Forest model.
    
    Parameters:
    - data: numpy array or pandas DataFrame, preprocessed data to fit the model.
    - contamination: float, the proportion of outliers in the data set.
    
    Returns:
    - clf: trained Isolation Forest model.
    """
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(data)
    
    # Save the Isolation Forest model after training
    save_isolation_forest_model(clf)
    
    return clf

def adapt_lstm(data, sequence_length, epochs=10, batch_size=32):
    """
    Adapts and fine-tunes an LSTM model.
    
    Parameters:
    - data: numpy array, preprocessed sequence data to fit the model.
    - sequence_length: int, the length of sequences for LSTM.
    - epochs: int, number of training epochs.
    - batch_size: int, size of the batches for training.
    
    Returns:
    - model: trained LSTM model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, data.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    return model

def prepare_sequences(data, sequence_length):
    """
    Prepares sequences from data for LSTM input.
    
    Parameters:
    - data: numpy array, input data.
    - sequence_length: int, length of each sequence.
    
    Returns:
    - X: numpy array, sequence input for LSTM.
    - y: numpy array, target values.
    """
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Ensemble Learning
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
    for iso_pred, lstm_pred in zip(isolation_forest_pred, lstm_pred):
        vote = (iso_pred + (1 if lstm_pred > threshold else 0)) / 2
        combined_pred.append(1 if vote >= 0.5 else -1)
    return np.array(combined_pred)

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

# Add this function to model_adaptation.py
def ensemble_model(isolation_forest_model, lstm_model, X_test):
    """
    Combines predictions from Isolation Forest and LSTM for ensemble decision-making.
    """
    # Get predictions from Isolation Forest
    isolation_forest_preds = isolation_forest_model.predict(X_test)
    
    # Prepare LSTM data
    X_seq, _ = prepare_sequences(X_test, sequence_length=5)
    
    # Get predictions from LSTM
    lstm_preds = lstm_model.predict(X_seq)
    
    # Ensemble decision (soft voting or averaging)
    ensemble_preds = (isolation_forest_preds + lstm_preds) / 2
    return ensemble_preds

# Crypto Analysis Integration
def evaluate_crypto_performance():
    from crypto_analysis import analyze_key_sizes, analyze_encapsulation_times, analyze_decapsulation_times
    
    key_sizes = analyze_key_sizes()
    encapsulation_times = analyze_encapsulation_times()
    decapsulation_times = analyze_decapsulation_times()

    print(f"Average Key Size: {np.mean(key_sizes)} bytes")
    print(f"Average Encapsulation Time: {np.mean(encapsulation_times)} seconds")
    print(f"Average Decapsulation Time: {np.mean(decapsulation_times)} seconds")

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = np.random.rand(1000, 10)  # 1000 samples, 10 features
    sequence_length = 5
    
    # Train and save Isolation Forest model
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    train_and_save_isolation_forest(X_train, 'Models/trained_models/trained_isolation_forest_model.pkl')

    # Prepare data for LSTM and train LSTM model
    X_seq, y_seq = prepare_sequences(data, sequence_length)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    train_and_save_lstm(X_train_seq, y_train_seq, sequence_length, 'Models/trained_models/trained_lstm_model.h5')
    
    # Save LSTM model after training
    save_lstm_model(lstm_model)
    
    # Ensemble evaluation
    ensemble_results = profile_code(evaluate_ensemble, X_test, y_test, isolation_forest_model, lstm_model, sequence_length)
    print("Ensemble model evaluation:", ensemble_results)

    # Evaluate ROC-AUC for the Isolation Forest model
    roc_auc = profile_code(evaluate_roc_auc, isolation_forest_model, X_test, y_test)
    print(f"Isolation Forest ROC-AUC: {roc_auc}")
    
    # Evaluate cryptographic performance
    profile_code(evaluate_crypto_performance)

    print("Model adaptation and evaluation complete.")
