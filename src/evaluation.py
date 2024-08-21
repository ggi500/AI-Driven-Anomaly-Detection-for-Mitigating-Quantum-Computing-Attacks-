from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def evaluate_isolation_forest(model, X_test, y_true):
    y_pred = model.predict(X_test)
    # Convert predictions to binary (1 for inliers, -1 for outliers)
    y_pred_binary = [1 if x == 1 else -1 for x in y_pred]
    
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_lstm(model, X_test, y_true):
    y_pred = model.predict(X_test)
    mse = np.mean((y_true - y_pred)**2)
    
    # You might want to set a threshold for anomaly detection
    threshold = np.mean(mse) + 2 * np.std(mse)
    y_pred_binary = [1 if x > threshold else -1 for x in mse]
    
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    return {"mse": mse, "precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    # Test your evaluation functions here
    # You'll need to load your models and test data first
    # isolation_forest_results = evaluate_isolation_forest(isolation_forest_model, X_test, y_test)
    # lstm_results = evaluate_lstm(lstm_model, X_test, y_test)
    print("Model evaluation complete.")