import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.model_adaptation import adapt_isolation_forest, load_model
from src.evaluation import evaluate_isolation_forest, evaluate_lstm, evaluate_model, evaluate_ensemble, evaluate_roc_auc
from src.generate_data import generate_and_save_data as generate_data
from adversarial_testing import evaluate_against_adversarial  # Import the adversarial testing function
from crypto_analysis import analyze_key_sizes, analyze_encapsulation_times, analyze_decapsulation_times, crypto_specific_metrics
from src.data_preprocessing import prepare_sequences
import tensorflow as tf
import logging
import keras_tuner as kt  # For automated hyperparameter tuning


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Manual Hyperparameter Tuning for LSTM Model
def build_model(hp):
    model = tf.keras.Sequential()
    
    # Tune the number of LSTM layers and units in each layer
    for i in range(hp.Int('num_layers', 1, 3)):  # Testing 1 to 3 layers
        model.add(tf.keras.layers.LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=256, step=32), 
                                       activation='relu',
                                       return_sequences=True if i < 2 else False,  # Only last layer doesn't return sequences
                                       input_shape=(None, hp.Int('input_dim', min_value=1, max_value=100))))
        model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(tf.keras.layers.Dense(1))  # Output layer
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), 
                  loss='mse')
    
    return model


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
    
    # Adding Cross-Validation (Optional)
    print("Applying K-fold Cross-validation...")
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Adapt Isolation Forest model
    print("Adapting Isolation Forest model...")
    isolation_forest_model = adapt_isolation_forest(X_train)

    # Use Keras Tuner for Automated Hyperparameter Tuning
    print("Running Hyperparameter Tuning with Keras Tuner...")
    tuner = kt.Hyperband(build_model,
                         objective='val_loss',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='lstm_tuning')

    # Perform the hyperparameter search using validation set
    tuner.search(X_train_fold, y_train_fold, epochs=10, validation_data=(X_val_fold, y_val_fold))

    # Get the best hyperparameters and build the model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    lstm_model = tuner.hypermodel.build(best_hps)

    # Train the model with the best hyperparameters
    lstm_model.fit(X_train_fold, y_train_fold, epochs=50, validation_data=(X_val_fold, y_val_fold))

    # Evaluate Isolation Forest model
    print("Evaluating Isolation Forest model...")
    isolation_forest_results = evaluate_isolation_forest(isolation_forest_model, X_test, y_test)
    roc_auc_if = evaluate_roc_auc(isolation_forest_model, X_test, y_test)

    # Evaluate LSTM model
    print("Evaluating LSTM model...")
    sequence_length = 10  # Adjust this based on your needs
    X_test_lstm = np.array([X_test[i:i+sequence_length] for i in range(len(X_test)-sequence_length)])
    y_test_lstm = y_test[sequence_length:]
    lstm_results = evaluate_lstm(lstm_model, X_test_lstm, y_test_lstm)
    roc_auc_lstm = evaluate_roc_auc(lstm_model, X_test_lstm, y_test_lstm)

    # Evaluate Ensemble
    print("Evaluating Ensemble model...")
    ensemble_results = evaluate_ensemble(X_test, y_test, isolation_forest_model, lstm_model, sequence_length)

    # Print results for all models
    print("\n---- Model Comparison ----")
    print(f"Isolation Forest: {isolation_forest_results}, ROC-AUC: {roc_auc_if}")
    print(f"LSTM: {lstm_results}, ROC-AUC: {roc_auc_lstm}")
    print(f"Ensemble: {ensemble_results}")

    # Add evaluation for ranking metrics (MAP, NDCG)
    print("\nEvaluating model with MAP and NDCG...")
    results = evaluate_model(lstm_model, X_test_lstm, y_test_lstm)
    
    # Print the evaluation results
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score}")

    #  Adversarial Testing  
    print("\nPerforming adversarial evaluation on the LSTM model...")
    epsilon = 0.01  # Perturbation amount for FGSM attack
    accuracy_adv, fpr_adv = evaluate_against_adversarial(lstm_model, X_test_lstm, y_test_lstm, epsilon=epsilon)

    # Print adversarial evaluation results
    print(f"\nAdversarial Accuracy: {accuracy_adv}")
    print(f"Adversarial False Positive Rate: {fpr_adv}")
    # --- Adversarial Testing Ends Here ---

    # Add basic data visualization
    if not os.path.exists('Data'):
        os.makedirs('Data')

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

    # CRYSTALS-Kyber-specific metrics
    print("\nAnalyzing CRYSTALS-Kyber operations...")

    # Collect key sizes, encapsulation times, and decapsulation times
    key_sizes = analyze_key_sizes()
    encapsulation_times = analyze_encapsulation_times()
    decapsulation_times = analyze_decapsulation_times()

    # Calculate and display custom metrics
    crypto_metrics = crypto_specific_metrics(key_sizes, encapsulation_times, decapsulation_times)
    print("\nCRYSTALS-Kyber Custom Metrics:")
    for metric, value in crypto_metrics.items():
        print(f"{metric}: {value}")

    print("Visualizations saved in the Data directory.")


if __name__ == "__main__":
    main()

