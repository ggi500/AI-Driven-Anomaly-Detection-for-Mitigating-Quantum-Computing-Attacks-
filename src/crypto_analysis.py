import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from pqcrypto.kem import kyber512

# Custom Metrics for CRYSTALS-Kyber
def crypto_specific_metrics(key_sizes, encapsulation_times, decapsulation_times):
    """
    Create custom metrics specific to CRYSTALS-Kyber implementation.
    
    Parameters:
    key_sizes: Array of key sizes
    encapsulation_times: Array of encapsulation times
    decapsulation_times: Array of decapsulation times
    
    Returns:
    dict: Custom metrics results
    """
    unexpected_key_size = np.sum(key_sizes != 4096) / len(key_sizes)
    abnormal_encapsulation = np.sum(encapsulation_times > 0.003) / len(encapsulation_times)
    decapsulation_failures = np.sum(decapsulation_times == 0) / len(decapsulation_times)
    
    return {
        "Unexpected Key Size Rate": unexpected_key_size,
        "Abnormal Encapsulation Time Rate": abnormal_encapsulation,
        "Decapsulation Failure Rate": decapsulation_failures
    }

# Kyber Operations Functions
def perform_kyber_operations():
    """
    Perform CRYSTALS-Kyber operations and measure encryption and decryption time.
    """
    public_key, secret_key = kyber512.keypair()

    # Measure encryption time
    start_time_enc = time.time()  # Start timer for encryption
    ciphertext, shared_secret_enc = kyber512.encrypt(public_key, np.random.bytes(32))
    end_time_enc = time.time()  # End timer
    encryption_time = end_time_enc - start_time_enc

    # Measure decryption time
    start_time_dec = time.time()  # Start timer for decryption
    shared_secret_dec = kyber512.decrypt(secret_key, ciphertext)
    end_time_dec = time.time()  # End timer
    decryption_time = end_time_dec - start_time_dec

    # Logging encryption and decryption times
    print(f"Encryption time: {encryption_time} seconds")
    print(f"Decryption time: {decryption_time} seconds")

    return {
        'public_key': public_key.hex(),
        'ciphertext': ciphertext.hex(),
        'shared_secret_enc': shared_secret_enc.hex(),
        'shared_secret_dec': shared_secret_dec.hex(),
        'key_size': len(public_key),
        'encryption_time': encryption_time,
        'decryption_time': decryption_time
    }

def analyze_key_sizes(n_samples=100):
    """
    Analyze the key sizes generated by CRYSTALS-Kyber.

    Parameters:
    n_samples (int): Number of key pairs to generate for analysis.

    Returns:
    np.array: Array containing the sizes of the generated public keys.
    """
    key_sizes = []
    for _ in range(n_samples):
        public_key, _ = kyber512.keypair()
        key_sizes.append(len(public_key))
    return np.array(key_sizes)

def analyze_encapsulation_times(n_samples=100):
    """
    Analyze the encapsulation times for CRYSTALS-Kyber.

    Parameters:
    n_samples (int): Number of encapsulation operations to perform for analysis.

    Returns:
    np.array: Array containing the encapsulation times in seconds.
    """
    encapsulation_times = []
    public_key, _ = kyber512.keypair()
    for _ in range(n_samples):
        start_time = time.time()
        _, _ = kyber512.encrypt(public_key)
        end_time = time.time()
        encapsulation_times.append(end_time - start_time)
    return np.array(encapsulation_times)

def analyze_decapsulation_times(n_samples=100):
    """
    Analyze the decapsulation times for CRYSTALS-Kyber.

    Parameters:
    n_samples (int): Number of decapsulation operations to perform for analysis.

    Returns:
    np.array: Array containing the decapsulation times in seconds.
    """
    decapsulation_times = []
    public_key, secret_key = kyber512.keypair()
    for _ in range(n_samples):
        ciphertext, _ = kyber512.encrypt(public_key)
        start_time = time.time()
        _ = kyber512.decrypt(secret_key, ciphertext)
        end_time = time.time()
        decapsulation_times.append(end_time - start_time)
    return np.array(decapsulation_times)

def preprocess_cryptographic_features():
    """
    Preprocess the cryptographic features by analyzing key sizes, encapsulation, and decapsulation times.

    Returns:
    np.array: Combined and normalized cryptographic features.
    """
    key_sizes = analyze_key_sizes()
    encapsulation_times = analyze_encapsulation_times()
    decapsulation_times = analyze_decapsulation_times()

    # Normalize the cryptographic features
    key_sizes = StandardScaler().fit_transform(np.array(key_sizes).reshape(-1, 1))
    encapsulation_times = StandardScaler().fit_transform(np.array(encapsulation_times).reshape(-1, 1))
    decapsulation_times = StandardScaler().fit_transform(np.array(decapsulation_times).reshape(-1, 1))

    # Combine cryptographic features into a feature array
    crypto_features = np.hstack((key_sizes, encapsulation_times, decapsulation_times))
    return crypto_features

if __name__ == "__main__":
    # Example usage of the cryptographic operations and analysis functions
    key_sizes = analyze_key_sizes()
    print(f"Average Key Size: {np.mean(key_sizes)} bytes")

    encapsulation_times = analyze_encapsulation_times()
    print(f"Average Encapsulation Time: {np.mean(encapsulation_times)} seconds")

    decapsulation_times = analyze_decapsulation_times()
    print(f"Average Decapsulation Time: {np.mean(decapsulation_times)} seconds")

    kyber_details = perform_kyber_operations()
    print(f"CRYSTALS-Kyber Operations: {kyber_details}")

    # Preprocess cryptographic features for anomaly detection
    crypto_features = preprocess_cryptographic_features()
    print("Preprocessed cryptographic features:")
    print(crypto_features)

    # Calculate custom metrics
    crypto_metrics = crypto_specific_metrics(key_sizes, encapsulation_times, decapsulation_times)
    print("\nCRYSTALS-Kyber Custom Metrics:")
    for metric, value in crypto_metrics.items():
        print(f"{metric}: {value}")
