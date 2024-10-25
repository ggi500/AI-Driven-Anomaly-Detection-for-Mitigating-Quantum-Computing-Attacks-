import numpy as np

# Key Management Anomaly Detection Rate (KMADR)
def calculate_kmadr(true_keys_anomalies, detected_keys_anomalies):
    """
    Calculate the Key Management Anomaly Detection Rate (KMADR).
    
    Parameters:
    true_keys_anomalies (array): True key-related anomalies (binary array where 1 = anomaly, 0 = normal)
    detected_keys_anomalies (array): Detected key-related anomalies (binary array)
    
    Returns:
    kmadr (float): Key Management Anomaly Detection Rate
    """
    tp_keys = np.sum((true_keys_anomalies == 1) & (detected_keys_anomalies == 1))  # True Positives
    fp_keys = np.sum((true_keys_anomalies == 0) & (detected_keys_anomalies == 1))  # False Positives
    fn_keys = np.sum((true_keys_anomalies == 1) & (detected_keys_anomalies == 0))  # False Negatives
    
    if (tp_keys + fp_keys + fn_keys) == 0:
        return 0.0  # Avoid division by zero
    
    kmadr = tp_keys / (tp_keys + fp_keys + fn_keys)
    return kmadr

# Encapsulation Time Outlier Detection Accuracy (ETODA)
def calculate_etoda(true_encap_anomalies, detected_encap_anomalies):
    """
    Calculate the Encapsulation Time Outlier Detection Accuracy (ETODA).
    
    Parameters:
    true_encap_anomalies (array): True encapsulation time anomalies (binary array)
    detected_encap_anomalies (array): Detected encapsulation time anomalies (binary array)
    
    Returns:
    etoda (float): Encapsulation Time Outlier Detection Accuracy
    """
    tp_encap = np.sum((true_encap_anomalies == 1) & (detected_encap_anomalies == 1))  # True Positives
    fp_encap = np.sum((true_encap_anomalies == 0) & (detected_encap_anomalies == 1))  # False Positives
    fn_encap = np.sum((true_encap_anomalies == 1) & (detected_encap_anomalies == 0))  # False Negatives
    
    if (tp_encap + fp_encap + fn_encap) == 0:
        return 0.0
    
    etoda = tp_encap / (tp_encap + fp_encap + fn_encap)
    return etoda

# Failed Decapsulation Event Rate (FDER)
def calculate_fder(failed_decap_events, total_decap_events):
    """
    Calculate the Failed Decapsulation Event Rate (FDER).
    
    Parameters:
    failed_decap_events (int): The number of failed decapsulation events
    total_decap_events (int): The total number of decapsulation events
    
    Returns:
    fder (float): Failed Decapsulation Event Rate
    """
    if total_decap_events == 0:
        return 0.0  # Avoid division by zero
    
    fder = failed_decap_events / total_decap_events
    return fder

# Post-Quantum Cryptanalysis Resistance Score (PQCRS)
def calculate_pqcrs(expected_attack_time, observed_decap_time):
    """
    Calculate the Post-Quantum Cryptanalysis Resistance Score (PQCRS).
    
    Parameters:
    expected_attack_time (float): Expected time to break the encryption using an attack (e.g., brute force)
    observed_decap_time (float): Observed decapsulation time during the cryptographic operation
    
    Returns:
    pqcrs (float): Post-Quantum Cryptanalysis Resistance Score
    """
    if observed_decap_time == 0:
        return 0.0  # Avoid division by zero or unrealistic scores
    
    pqcrs = expected_attack_time / observed_decap_time
    return pqcrs

# Quantum Cryptography Anomaly Detection F1-Score
def calculate_f1_score(true_anomalies, detected_anomalies):
    """
    Calculate the F1-score for quantum cryptography anomaly detection.
    
    Parameters:
    true_anomalies (array): True anomalies (binary array)
    detected_anomalies (array): Detected anomalies (binary array)
    
    Returns:
    f1_score (float): F1-score for anomaly detection
    """
    tp = np.sum((true_anomalies == 1) & (detected_anomalies == 1))  # True Positives
    fp = np.sum((true_anomalies == 0) & (detected_anomalies == 1))  # False Positives
    fn = np.sum((true_anomalies == 1) & (detected_anomalies == 0))  # False Negatives
    
    if tp + fp == 0 or tp + fn == 0:
        return 0.0  # Avoid division by zero
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    if precision + recall == 0:
        return 0.0  # Avoid division by zero
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score
