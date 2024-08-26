import kaggle
import os

# Set the path where you want to download the data
data_dir = 'Data/UNSW-NB15'
os.makedirs(data_dir, exist_ok=True)

# Download the dataset
# Replace 'mrwellsdavid/unsw-nb15' with the actual dataset identifier if different
kaggle.api.dataset_download_files('mrwellsdavid/unsw-nb15', path=data_dir, unzip=True)

print(f"Dataset downloaded and extracted to {data_dir}")