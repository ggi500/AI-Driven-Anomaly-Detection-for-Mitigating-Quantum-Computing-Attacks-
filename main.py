from src.data_preprocessing import preprocess_data, load_ieee_cis_data, add_kyber_elements, preprocess_pipeline, load_datasets
from src.model_adaptation import adapt_isolation_forest, adapt_lstm
from src.evaluation import evaluate_isolation_forest, evaluate_lstm
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_pipeline import main as generate_data

def main():
    print("Current working directory:", os.getcwd())
    
    # Ensure Data directory exists
    os.makedirs('Data', exist_ok=True)
    print("Files in Data directory:", os.listdir('Data'))

    data_file = 'Data/processed_swift_data.csv'
    
    if not os.path.exists(data_file):
        print(f"File {data_file} not found. Generating data...")
        generate_data()
    
    if os.path.exists(data_file):
        print(f"Loading data from {data_file}")
        swift_data = pd.read_csv(data_file)
        time_series_data = np.load('Data/time_series_data.npy')
    else:
        print(f"Error: {data_file} still doesn't exist after attempting to generate it.")
        return

    print("Data preprocessing complete.")
    print("Shape of preprocessed SWIFT-like data:", swift_data.shape)
    print("Shape of time series data:", time_series_data.shape)

    # Save the processed data to the Data directory (in case it wasn't saved during generation)
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

    # Add basic data visualization
    print("\nGenerating basic visualizations...")
    
    # Histogram of transaction amounts
    plt.figure(figsize=(10, 6))
    sns.histplot(swift_data['amount'], bins=50)
    plt.title('Distribution of Transaction Amounts')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.savefig('Data/transaction_amounts_distribution.png')
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(swift_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Features')
    plt.savefig('Data/correlation_heatmap.png')
    plt.close()

    print("Visualizations saved in the Data directory.")

if __name__ == "__main__":
    main()