import lime
import lime.lime_tabular
import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

def explain_instance_with_lime(model, X_sample, feature_names, model_type='LSTM'):
    """
    Generate LIME explanations for a single instance from the test set.
    
    Parameters:
    - model: Trained model (LSTM or Isolation Forest)
    - X_sample: Single test instance
    - feature_names: List of feature names
    - model_type: 'LSTM' or 'Isolation Forest'
    
    Returns:
    - explanation: LIME explanation object
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(X_sample, feature_names=feature_names, class_names=['Anomaly', 'Normal'],
                                                       discretize_continuous=True)
    
    if model_type == 'LSTM':
        # LSTM uses sequence data, so we'll need to flatten the sequence
        X_sample = X_sample.flatten()
    
    explanation = explainer.explain_instance(X_sample, model.predict if model_type == 'LSTM' else model.decision_function)
    explanation.show_in_notebook()
    return explanation

def shap_explanation(model, X_train, y_train):
    """
    Generate SHAP explanations for a model.
    
    Parameters:
    - model: Trained model (XGBoost or other tree-based model)
    - X_train: Training data features
    - y_train: Training data labels
    
    Returns:
    - shap_values: SHAP values for the model
    """
    # If the model is not trained, train it
    if not hasattr(model, 'feature_importances_'):
        model.fit(X_train, y_train)
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Plot SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    plt.close()
    
    # Plot SHAP force plot for the first prediction
    plt.figure(figsize=(12, 3))
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:], matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig('shap_force_plot.png')
    plt.close()

    return shap_values

# Example usage
if __name__ == "__main__":
    # Load your data and models here
    from data_preprocessing import load_processed_data
    from model_adaptation import load_lstm_model, load_isolation_forest_model
    
    X_test, y_test, feature_names = load_processed_data()
    lstm_model = load_lstm_model()
    isolation_forest_model = load_isolation_forest_model()
    
    # LIME explanation
    sample_index = 0  # You can adjust this
    lime_explanation_lstm = explain_instance_with_lime(lstm_model, X_test[sample_index], feature_names, model_type='LSTM')
    lime_explanation_if = explain_instance_with_lime(isolation_forest_model, X_test[sample_index], feature_names, model_type='Isolation Forest')
    
    # SHAP explanation (assuming you have an XGBoost model)
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_test, y_test)  # Train the model if not already trained
    shap_values = shap_explanation(xgb_model, X_test, y_test)
    
    print("Model explanations completed. Check the generated LIME and SHAP visualizations.")