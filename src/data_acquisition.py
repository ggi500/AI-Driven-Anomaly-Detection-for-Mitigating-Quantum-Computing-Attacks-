import os
import requests
import zipfile
import kaggle

# Create directories for datasets
os.makedirs('datasets/unsw_nb15', exist_ok=True)
os.makedirs('datasets/cic_ids2017', exist_ok=True)
os.makedirs('datasets/ieee_cis_fraud', exist_ok=True)
os.makedirs('datasets/paysim', exist_ok=True)

# Function to download a file from a URL
def download_file(url, dest_folder, dest_name=None):
    if dest_name is None:
        dest_name = url.split("/")[-1]
    dest_path = os.path.join(dest_folder, dest_name)
    if not os.path.exists(dest_path):
        print(f"Downloading {dest_name}...")
        response = requests.get(url, stream=True)
        with open(dest_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {dest_name} to {dest_path}")
    else:
        print(f"{dest_name} already exists, skipping download.")
    return dest_path

# Function to extract a zip file
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

# UNSW-NB15
def download_unsw_nb15():
    url = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
    file_path = download_file(url, "datasets/unsw_nb15", "UNSW-NB15.zip")
    extract_zip(file_path, "datasets/unsw_nb15")

# CIC-IDS2017
def download_cic_ids2017():
    # Note: CIC-IDS2017 dataset might be large and split into multiple files.
    # Example URL - replace with the correct link if necessary.
    url = "https://www.unb.ca/cic/datasets/ids-2017.html"
    file_path = download_file(url, "datasets/cic_ids2017", "CIC-IDS2017.zip")
    extract_zip(file_path, "datasets/cic_ids2017")

# IEEE-CIS Fraud Detection (Kaggle)
def download_ieee_cis_fraud():
    kaggle.api.authenticate()
    kaggle.api.competition_download_files('ieee-fraud-detection', path='datasets/ieee_cis_fraud', quiet=False)
    extract_zip('datasets/ieee_cis_fraud/ieee-fraud-detection.zip', 'datasets/ieee_cis_fraud')

# PaySim (Kaggle)
def download_paysim():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('ntnu-testimon/paysim1', path='datasets/paysim', quiet=False, unzip=True)

# Download and load datasets
download_unsw_nb15()
download_cic_ids2017()
download_ieee_cis_fraud()
download_paysim()

print("All datasets downloaded and extracted.")

# Load datasets using pandas
import pandas as pd

# Example paths - adjust according to your actual files
def load_datasets():
    unsw_nb15_data = pd.read_csv('datasets/unsw_nb15/UNSW-NB15.csv')  # Replace with actual file path
    cic_ids2017_data = pd.read_csv('datasets/cic_ids2017/CIC-IDS2017.csv')  # Replace with actual file path
    ieee_cis_fraud_data = pd.read_csv('datasets/ieee_cis_fraud/train_transaction.csv')
    paysim_data = pd.read_csv('datasets/paysim/PS_20174392719_1491204439457_log.csv')
    
    return unsw_nb15_data, cic_ids2017_data, ieee_cis_fraud_data, paysim_data

# Load the datasets
unsw_nb15_data, cic_ids2017_data, ieee_cis_fraud_data, paysim_data = load_datasets()

print("Datasets loaded into memory.")