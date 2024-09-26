import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.model_adaptation import adapt_isolation_forest, adapt_lstm
from src.evaluation import evaluate_isolation_forest, evaluate_lstm
from src.generate_data import generate_and_save_data as generate_data
from src.evaluation import evaluate_model  # Add this to import the evaluation function for MAP, NDCG, etc.

def main():
    print("Generating or loading data...")
    swift_data, time_series_data = generate_data()  # Using the generate_data function to load/generate data
    
    if swift_data is None or time_series_data is None:
        print("Error: Failed to generate or load data.")
        return

    print("Data loaded successfully.")
    print("Shape of SWIFT-like data:", swift_data.shape)
    print("Shape of time series data:", time_series_data.shape)

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

    # Print results for Isolation Forest and LSTM
    print("\nIsolation Forest Results:")
    for metric, value in isolation_forest_results.items():
        print(f"{metric}: {value}")

    print("\nLSTM Results:")
    for metric, value in lstm_results.items():
        print(f"{metric}: {value}")

    # Add evaluation for ranking metrics (MAP, NDCG)
    print("\nEvaluating model with MAP and NDCG...")
    results = evaluate_model(lstm_model, X_test_lstm, y_test_lstm)  # This is where you add the evaluation
    
    # Print the evaluation results
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score}")

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
