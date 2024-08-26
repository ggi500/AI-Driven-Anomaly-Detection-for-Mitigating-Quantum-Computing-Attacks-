import kaggle
import os

# Define the directory to save the dataset
data_dir = 'Data/UNSW-NB15'
os.makedirs(data_dir, exist_ok=True)

# Download and unzip the dataset
kaggle.api.dataset_download_files('mrwellsdavid/unsw-nb15', path=data_dir, unzip=True)

print(f"Dataset downloaded and extracted to {data_dir}")
