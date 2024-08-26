import kaggle
import os

# List datasets
datasets = kaggle.api.dataset_list()
for dataset in datasets[:5]:  # Print first 5 datasets
    print(dataset)