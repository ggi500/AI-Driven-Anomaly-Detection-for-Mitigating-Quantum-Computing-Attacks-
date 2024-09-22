import joblib
import tensorflow as tf

# Function to save Isolation Forest
def train_and_save_isolation_forest(X_train, model_save_path):
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    # Save model to `trained_models` directory
    joblib.dump(model, model_save_path)
    print(f"Isolation Forest model saved at {model_save_path}")

# Function to save LSTM model
def train_and_save_lstm(X_train, y_train, sequence_length, model_save_path):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(sequence_length, X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save model to `trained_models` directory
    model.save(model_save_path)
    print(f"LSTM model saved at {model_save_path}")
