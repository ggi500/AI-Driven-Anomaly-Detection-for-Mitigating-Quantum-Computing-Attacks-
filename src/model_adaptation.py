import tensorflow as tf
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

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

def train_and_evaluate_lstm(data, sequence_length, test_size=0.2, epochs=10, batch_size=32):
    """
    Trains and evaluates an LSTM model.
    
    Parameters:
    - data: numpy array, preprocessed sequence data.
    - sequence_length: int, length of sequences.
    - test_size: float, proportion of the data to use for testing.
    - epochs: int, number of training epochs.
    - batch_size: int, size of the batches for training.
    
    Returns:
    - model: trained LSTM model.
    - evaluation metrics: accuracy, F1-score, AUC, and confusion matrix.
    """
    X, y = prepare_sequences(data, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = adapt_lstm(X_train, sequence_length, epochs=epochs, batch_size=batch_size)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)  # Round predictions to 0 or 1
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, f1, auc, cm

if __name__ == "__main__":
    # Example usage:
    # Simulating some data for demonstration
    np.random.seed(42)
    data = np.random.rand(1000, 10)  # 1000 samples, 10 features
    sequence_length = 5
    
    # Isolation Forest
    isolation_forest_model = adapt_isolation_forest(data)
    print("Isolation Forest model adapted")

    # LSTM
    model, accuracy, f1, auc, cm = train_and_evaluate_lstm(data, sequence_length)
    print(f"LSTM model adapted with Accuracy: {accuracy}, F1-score: {f1}, AUC: {auc}")
    print("Confusion Matrix:\n", cm)

    print("Model adaptation complete.")
