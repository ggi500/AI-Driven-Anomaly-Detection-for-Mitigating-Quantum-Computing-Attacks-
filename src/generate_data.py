from data_preprocessing import preprocess_data, load_ieee_cis_data, add_kyber_elements, preprocess_pipeline, load_datasets
import pandas as pd
import numpy as np
import os

def generate_and_save_data():
    print("Generating data...")
    # Ensure Data directory exists
    os.makedirs('Data', exist_ok=True)

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

if __name__ == "__main__":
    generate_and_save_data()