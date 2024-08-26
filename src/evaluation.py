import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from sklearn.utils import resample

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

# Evaluation Functions
def evaluate_isolation_forest(model, X_test, y_true):
    """
    Evaluates the performance of the Isolation Forest model using custom metrics.
    
    Parameters:
    - model: trained Isolation Forest model.
    - X_test: numpy array, test data.
    - y_true: numpy array, true labels.
    
    Returns:
    - results: dict, contains precision, recall, and F1-score.
    """
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if x == 1 else -1 for x in y_pred]
    
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_lstm(model, X_test, y_true):
    """
    Evaluates the performance of the LSTM model using custom metrics.
    
    Parameters:
    - model: trained LSTM model.
    - X_test: numpy array, test data.
    - y_true: numpy array, true labels.
    
    Returns:
    - results: dict, contains MSE, precision, recall, and F1-score.
    """
    y_pred = model.predict(X_test)
    mse = np.mean((y_true - y_pred)**2)
    
    # Set a threshold for anomaly detection
    threshold = np.mean(mse) + 2 * np.std(mse)
    y_pred_binary = [1 if x > threshold else -1 for x in mse]
    
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    return {"mse": mse, "precision": precision, "recall": recall, "f1": f1}

def k_fold_cross_validation(model_func, X, y, k=5, sequence_length=None):
    """
    Performs k-fold cross-validation on the given model.
    
    Parameters:
    - model_func: function, the model creation function (e.g., adapt_lstm, adapt_isolation_forest).
    - X: numpy array, input data.
    - y: numpy array, target labels.
    - k: int, number of folds.
    - sequence_length: int, the length of sequences (for LSTM).
    
    Returns:
    - avg_metrics: dict, average metrics across all folds.
    """
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
    return avg_metrics

def bootstrap_sampling(model_func, X, y, n_iterations=100, sequence_length=None):
    """
    Performs bootstrap sampling to estimate the model performance.
    
    Parameters:
    - model_func: function, the model creation function (e.g., adapt_lstm, adapt_isolation_forest).
    - X: numpy array, input data.
    - y: numpy array, target labels.
    - n_iterations: int, number of bootstrap samples.
    - sequence_length: int, the length of sequences (for LSTM).
    
    Returns:
    - avg_metrics: dict, average metrics across all iterations.
    """
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
    return avg_metrics

if __name__ == "__main__":
    # Example usage:
    # Simulating some data for demonstration
    np.random.seed(42)
    data = np.random.rand(1000, 10)  # 1000 samples, 10 features
    sequence_length = 5

    # Isolation Forest
    isolation_forest_model = adapt_isolation_forest(data)
    X_train, X_test, y_train, y_test = train_test_split(data, np.random.randint(0, 2, size=1000), test_size=0.2, random_state=42)
    isolation_forest_results = evaluate_isolation_forest(isolation_forest_model, X_test, y_test)
    print("Isolation Forest model evaluation:", isolation_forest_results)

    # LSTM
    X_seq, y_seq = prepare_sequences(data, sequence_length)
    lstm_model = adapt_lstm(X_seq, sequence_length)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32)
    lstm_results = evaluate_lstm(lstm_model, X_test_seq, y_test_seq)
    print("LSTM model evaluation:", lstm_results)

    # K-fold Cross-Validation
    k_fold_metrics = k_fold_cross_validation(adapt_lstm, data, np.random.randint(0, 2, size=1000), k=5, sequence_length=sequence_length)
    print("K-fold Cross-Validation metrics:", k_fold_metrics)

    # Bootstrap Sampling
    bootstrap_metrics = bootstrap_sampling(adapt_lstm, data, np.random.randint(0, 2, size=1000), n_iterations=100, sequence_length=sequence_length)
    print("Bootstrap Sampling metrics:", bootstrap_metrics)

    print("Model adaptation and evaluation complete.")
