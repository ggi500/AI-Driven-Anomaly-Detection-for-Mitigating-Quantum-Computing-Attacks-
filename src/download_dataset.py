import os
import requests
import zipfile
import kaggle
import pandas as pd

# Create directories for datasets
os.makedirs('datasets/unsw_nb15', exist_ok=True)
os.makedirs('datasets/cic_ids2017', exist_ok=True)
os.makedirs('datasets/ieee_cis_fraud', exist_ok=True)
os.makedirs('datasets/paysim', exist_ok=True)

# Function to download a dataset from Kaggle
def download_kaggle_dataset(dataset_name, data_dir):
    if not os.listdir(data_dir):  # Download only if the directory is empty
        print(f"Downloading {dataset_name} dataset...")
        kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
        print(f"{dataset_name} dataset downloaded and extracted to {data_dir}")
    else:
        print(f"{dataset_name} dataset already exists in {data_dir}, skipping download.")

# Download datasets from Kaggle
def download_datasets():
    download_kaggle_dataset('mrwellsdavid/unsw-nb15', 'datasets/unsw_nb15')
    download_kaggle_dataset('devendra416/cicids2017', 'datasets/cic_ids2017')
    download_kaggle_dataset('ntnu-testimon/paysim1', 'datasets/paysim')

    # IEEE-CIS Fraud Detection requires authentication and is larger, so it's separate
    if not os.listdir('datasets/ieee_cis_fraud'):
        print("Downloading IEEE-CIS Fraud Detection dataset...")
        kaggle.api.competition_download_files('ieee-fraud-detection', path='datasets/ieee_cis_fraud', quiet=False)
        print("IEEE-CIS Fraud Detection dataset downloaded.")
    else:
        print("IEEE-CIS Fraud Detection dataset already exists in 'datasets/ieee_cis_fraud', skipping download.")

# Load datasets using pandas
def load_datasets():
    unsw_nb15_data = pd.read_csv('datasets/unsw_nb15/UNSW-NB15.csv')  
    cic_ids2017_data = pd.read_csv('datasets/cic_ids2017/CIC-IDS2017.csv')  
    ieee_cis_fraud_data = pd.read_csv('datasets/ieee_cis_fraud/train_transaction.csv')
    paysim_data = pd.read_csv('datasets/paysim/PS_20174392719_1491204439457_log.csv')
    
    return unsw_nb15_data, cic_ids2017_data, ieee_cis_fraud_data, paysim_data

# Run the download function
download_datasets()

# Load the datasets
unsw_nb15_data, cic_ids2017_data, ieee_cis_fraud_data, paysim_data = load_datasets()

print("Datasets loaded into memory.")
