import os
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_pipeline, load_datasets
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # Importing SMOTE

# Check if essential columns exist in the dataset
def validate_swift_dataset(dataset, required_columns):
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"The dataset is missing essential columns: {missing_columns}")
    else:
        print(f"All required columns are present: {required_columns}")

def generate_and_save_data():
    print("Generating or loading data...")
    
    # Ensure Data directory exists
    data_dir = os.path.join(os.getcwd(), 'Data')
    os.makedirs(data_dir, exist_ok=True)

    # Define file paths
    swift_data_path = os.path.join(data_dir, 'processed_swift_data.csv')
    time_series_data_path = os.path.join(data_dir, 'time_series_data.npy')

    # Check if the files already exist
    if os.path.exists(swift_data_path) and os.path.exists(time_series_data_path):
        try:
            # Load the existing data
            swift_data = pd.read_csv(swift_data_path)
            time_series_data = np.load(time_series_data_path)
            print("Data loaded successfully from existing files.")
        except Exception as e:
            print(f"An error occurred while loading the files: {str(e)}")
            return None, None
    else:
        try:
            # Load and preprocess multiple datasets
            print("Loading and preprocessing data...")
            unsw, cicids, ieee_cis, paysim = load_datasets()
            swift_data, time_series_data = preprocess_pipeline(unsw, cicids, ieee_cis, paysim)
            
            # Validate that essential columns are present
            required_columns = ['amount', 'currency', 'timestamp']
            validate_swift_dataset(swift_data, required_columns)
            
            print("Data preprocessing complete.")
            print("Shape of preprocessed SWIFT-like data:", swift_data.shape)
            print("Shape of time series data:", time_series_data.shape)

            # Apply SMOTE to handle class imbalance before saving the data
            X = swift_data.iloc[:, :-1].values  # Features
            y = swift_data.iloc[:, -1].values   # Labels

            # Splitting data into training and test sets (before applying SMOTE)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Applying SMOTE to the training set only
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print("SMOTE applied to handle class imbalance on training data.")

            # Combine the resampled training data and the test data if you want to save the full dataset
            X_resampled = np.vstack([X_train_resampled, X_test])
            y_resampled = np.hstack([y_train_resampled, y_test])
            swift_data_resampled = pd.DataFrame(np.column_stack([X_resampled, y_resampled]), columns=swift_data.columns)

            # Save the processed and resampled data to the Data directory
            swift_data_resampled.to_csv(swift_data_path, index=False)
            np.save(time_series_data_path, time_series_data)
            
            print(f"Processed and resampled data saved to {swift_data_path}")
            print(f"Time series data saved to {time_series_data_path}")
        except Exception as e:
            print(f"An error occurred during data generation: {str(e)}")
            return None, None

    return swift_data, time_series_data

if __name__ == "__main__":
    generate_and_save_data()