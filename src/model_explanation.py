import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

def shap_explanation(model, X_train, y_train):
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
    # Load some example data
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = [0, 1, 0]
    
    # Train an example model
    model = xgb.XGBClassifier()
    
    # Run SHAP explanation
    shap_values = shap_explanation(model, X_train, y_train)
    print("SHAP explanation completed. Check the generated plot images.")