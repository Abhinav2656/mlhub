# core/utils.py
import pickle

def load_model(model_name):
    """
    Load a pre-trained model from the ml_models directory.
    model_name: a string (e.g., 'logistic_model' or 'linear_regression_model')
    """
    model_path = f"ml_models/{model_name}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def run_prediction(model, data):
    """
    Run prediction using the loaded model on the provided DataFrame.
    Returns the predictions.
    """
    predictions = model.predict(data)
    return predictions
