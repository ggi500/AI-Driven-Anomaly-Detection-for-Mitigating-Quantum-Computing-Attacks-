import os
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FGSM Attack for TensorFlow Models
def fgsm_attack(model, data, labels, epsilon=0.01):
    """
    Perform FGSM attack on a TensorFlow model.
    
    Parameters:
    model: The model to attack
    data: Input data to perturb
    labels: True labels of the data
    epsilon: Amount of perturbation
    
    Returns:
    adversarial_data: Perturbed data
    """
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels)
    
    with tf.GradientTape() as tape:
        tape.watch(data)
        predictions = model(data)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, predictions)
    
    gradient = tape.gradient(loss, data)
    signed_grad = tf.sign(gradient)
    perturbed_data = data + epsilon * signed_grad
    perturbed_data = tf.clip_by_value(perturbed_data, 0, 1)
    
    return perturbed_data.numpy()

# FGSM Attack for PyTorch Models
def fgsm_attack_torch(model, data, labels, epsilon=0.01):
    """
    Perform FGSM attack on a PyTorch model.
    
    Parameters:
    model: The model to attack
    data: Input data to perturb
    labels: True labels of the data
    epsilon: Amount of perturbation
    
    Returns:
    adversarial_data: Perturbed data
    """
    model.eval()
    data.requires_grad = True
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, labels)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

# Generate Adversarial Examples (Gradient-Free Method)
def generate_adversarial_examples(model, data):
    """
    Example adversarial generation method using random noise.
    """
    adv_data = data + 0.01 * np.sign(np.random.randn(*data.shape))
    return adv_data

# Evaluation Against Adversarial Examples
def evaluate_against_adversarial(model, data, labels, attack_type='random', epsilon=0.01, use_torch=False):
    """
    Evaluates the model against adversarial examples.
    
    Parameters:
    model: The model to evaluate
    data: Input data to perturb
    labels: True labels
    attack_type: Type of attack ('random' or 'fgsm')
    epsilon: Perturbation size for FGSM attack
    use_torch: Whether the model is PyTorch-based or TensorFlow-based
    
    Returns:
    accuracy: Model accuracy against adversarial examples
    fpr: False positive rate against adversarial examples
    """
    if attack_type == 'fgsm':
        if use_torch:
            data_tensor = torch.tensor(data, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            adv_data = fgsm_attack_torch(model, data_tensor, labels_tensor, epsilon).detach().numpy()
        else:
            adv_data = fgsm_attack(model, data, labels, epsilon)
    else:
        adv_data = generate_adversarial_examples(model, data)

    preds = model.predict(adv_data)
    accuracy = np.mean(preds == labels)
    fpr = false_positive_rate(labels, preds)

    logger.info(f"Adversarial Accuracy: {accuracy}")
    logger.info(f"False Positive Rate: {fpr}")
    
    return accuracy, fpr

# Calculate False Positive Rate (FPR)
def false_positive_rate(y_true, y_pred):
    """
    Calculates False Positive Rate (FPR).
    """
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1]
    tn = cm[0, 0]
    return fp / (fp + tn)

# New function for running iterative adversarial tests
def run_adversarial_tests(model, X_test, y_test, model_type='tensorflow', epsilon=0.01, iterations=10):
    """
    Runs adversarial testing on a model and logs performance across iterations.
    
    Parameters:
    - model: Trained model (TensorFlow or PyTorch)
    - X_test: Test data
    - y_test: Test labels
    - model_type: Type of model ('tensorflow' or 'pytorch')
    - epsilon: Perturbation amount for adversarial attacks
    - iterations: Number of adversarial testing iterations
    
    Returns:
    - results: Adversarial test results (accuracy, FPR)
    """
    results = {'accuracy': [], 'fpr': []}
    
    for i in range(iterations):
        logger.info(f"Running adversarial test iteration {i + 1}/{iterations} for {model_type}...")
        
        if model_type == 'tensorflow':
            adv_data = fgsm_attack(model, X_test, y_test, epsilon)
        elif model_type == 'pytorch':
            data_tensor = torch.tensor(X_test, dtype=torch.float32)
            labels_tensor = torch.tensor(y_test, dtype=torch.long)
            adv_data = fgsm_attack_torch(model, data_tensor, labels_tensor, epsilon).detach().numpy()
        else:
            raise ValueError("Invalid model_type. Use 'tensorflow' or 'pytorch'.")
        
        # Predict on perturbed data
        if model_type == 'tensorflow':
            preds = model.predict(adv_data)
        else:  # PyTorch
            with torch.no_grad():
                preds = model(torch.tensor(adv_data, dtype=torch.float32)).argmax(dim=1).numpy()
        
        accuracy = np.mean(preds == y_test)
        fpr = false_positive_rate(y_test, preds)
        
        results['accuracy'].append(accuracy)
        results['fpr'].append(fpr)
        
        logger.info(f"Iteration {i + 1} - Accuracy: {accuracy:.4f}, FPR: {fpr:.4f}")
    
    return results

# Example for Model Evaluation and Adversarial Testing
if __name__ == "__main__":
    # Example of loading a trained model (TensorFlow/Keras or PyTorch)
    # Use the corresponding loading function depending on the framework.

    # For TensorFlow/Keras:
    model = tf.keras.models.load_model('path_to_tensorflow_model.h5')
    model_type = 'tensorflow'

    # For PyTorch:
    # model = torch.load('path_to_pytorch_model.pth')
    # model_type = 'pytorch'
    
    # Load test dataset
    X_test = np.load('X_test.npy')  # Load test features
    y_test = np.load('y_test.npy')  # Load test labels

    # Perform single adversarial test using FGSM attack
    epsilon = 0.02  # Perturbation size for FGSM attack
    accuracy, fpr = evaluate_against_adversarial(model, X_test, y_test, attack_type='fgsm', epsilon=epsilon, use_torch=(model_type=='pytorch'))
    
    logger.info("Single adversarial evaluation complete.")

    # Run iterative adversarial tests
    results = run_adversarial_tests(model, X_test, y_test, model_type=model_type, epsilon=epsilon, iterations=10)
    
    logger.info("Iterative Adversarial Test Results:")
    logger.info(f"Average Accuracy: {np.mean(results['accuracy']):.4f}")
    logger.info(f"Average FPR: {np.mean(results['fpr']):.4f}")

    logger.info("Adversarial testing complete.")