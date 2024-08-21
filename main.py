from src.data_preprocessing import preprocess_data, load_ieee_cis_data, add_kyber_elements, preprocess_pipeline, load_datasets
from src.model_adaptation import adapt_isolation_forest, adapt_lstm
from src.evaluation import evaluate_isolation_forest, evaluate_lstm
from sklearn.model_selection import train_test_split
import numpy as np
import os

def main():
    # Check if the Data directory exists
    if not os.path.exists('Data'):
        print("Data directory does not exist.")
        os.makedirs('Data')
        print("Data directory created.")
    else:
        # List files in the Data directory
        print("Files in Data directory:", os.listdir('Data'))

        # Check if the processed_swift_data.csv file exists
        if 'processed_swift_data.csv' in os.listdir('Data'):
            print("processed_swift_data.csv exists.")
        else:
            print("processed_swift_data.csv does not exist.")

    # Load and preprocess multiple datasets
    print("Loading and preprocessing data...")
    unsw, cicids, ieee_cis, paysim = load_datasets()
    swift_data, time_series_data = preprocess_pipeline(unsw, cicids, ieee_cis, paysim)
    
    print("Data preprocessing complete.")
    print("Shape of preprocessed SWIFT-like data:", swift_data.shape)
    print("Shape of time series data:", time_series_data.shape)
    
    # Save the processed data to the Data directory
    swift_data.to_csv('Data/processed_swift_data.csv', index=False)
    np.save('Data/time_series_data.npy', time_series_data)
    
    print("Processed data saved to Data/processed_swift_data.csv")

    # Final check for the file's existence (optional)
    if os.path.exists('Data/processed_swift_data.csv'):
        print("The file 'processed_swift_data.csv' has been created successfully!")
    else:
        print("The file 'processed_swift_data.csv' was not created.")

    # Split data into features (X) and target (y)
    print("Splitting data...")
    X = swift_data.iloc[:, :-1].values
    y = swift_data.iloc[:, -1].values

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Adapt Isolation Forest model
    print("Adapting Isolation Forest model...")
    isolation_forest_model = adapt_isolation_forest(X_train)
    
    # Adapt LSTM model
    print("Adapting LSTM model...")
    sequence_length = 10  # Adjust this based on your needs
    X_train_lstm = np.array([X_train[i:i+sequence_length] for i in range(len(X_train)-sequence_length)])
    y_train_lstm = y_train[sequence_length:]
    lstm_model = adapt_lstm(X_train_lstm, sequence_length)

    # Evaluate Isolation Forest model
    print("Evaluating Isolation Forest model...")
    isolation_forest_results = evaluate_isolation_forest(isolation_forest_model, X_test, y_test)
    
    # Evaluate LSTM model
    print("Evaluating LSTM model...")
    X_test_lstm = np.array([X_test[i:i+sequence_length] for i in range(len(X_test)-sequence_length)])
    y_test_lstm = y_test[sequence_length:]
    lstm_results = evaluate_lstm(lstm_model, X_test_lstm, y_test_lstm)

    # Print results
    print("\nIsolation Forest Results:")
    for metric, value in isolation_forest_results.items():
        print(f"{metric}: {value}")

    print("\nLSTM Results:")
    for metric, value in lstm_results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
