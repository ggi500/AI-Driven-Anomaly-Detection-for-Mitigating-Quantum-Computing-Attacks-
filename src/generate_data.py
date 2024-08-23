# generate_data.py
import os
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_pipeline, load_datasets
def generate_and_save_data():
    print("Generating data...")
    
    # Ensure Data directory exists
    data_dir = os.path.join(os.getcwd(), 'Data')
    os.makedirs(data_dir, exist_ok=True)

    try:
        # Load and preprocess multiple datasets
        print("Loading and preprocessing data...")
        unsw, cicids, ieee_cis, paysim = load_datasets()
        swift_data, time_series_data = preprocess_pipeline(unsw, cicids, ieee_cis, paysim)
        
        print("Data preprocessing complete.")
        print("Shape of preprocessed SWIFT-like data:", swift_data.shape)
        print("Shape of time series data:", time_series_data.shape)
        
        # Save the processed data to the Data directory
        swift_data_path = os.path.join(data_dir, 'processed_swift_data.csv')
        time_series_data_path = os.path.join(data_dir, 'time_series_data.npy')
        
        swift_data.to_csv(swift_data_path, index=False)
        np.save(time_series_data_path, time_series_data)
        
        # Verify files were created
        if os.path.exists(swift_data_path) and os.path.exists(time_series_data_path):
            print(f"Processed data saved to {swift_data_path}")
            print(f"Time series data saved to {time_series_data_path}")
        else:
            print("Error: Files were not created successfully.")
    
    except Exception as e:
        print(f"An error occurred during data generation: {str(e)}")
