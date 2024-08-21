import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "tensorflow",
    "keras",
    "matplotlib",
    "seaborn",
    "pqcrypto",
    "faker"
]

# Install packages
for package in packages:
    install(package)

# Verify installations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from pqcrypto.kem import kyber512
from faker import Faker

print("All required packages are installed and imported successfully.")

# Check TensorFlow GPU support
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.test.is_built_with_cuda() else "NOT AVAILABLE")

# Verify Scikit-learn version
import sklearn
print("Scikit-learn version:", sklearn.__version__)

# Test CRYSTALS-Kyber
public_key, secret_key = kyber512.keypair()
print("CRYSTALS-Kyber test successful")

print("Environment setup complete.") 