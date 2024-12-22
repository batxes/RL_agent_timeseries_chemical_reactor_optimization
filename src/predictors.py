import numpy as np
import logging

class DummyPredictor:
    """Fallback predictor when LSTM model fails to load"""
    def __init__(self):
        logging.warning("Using DummyPredictor - predictions will be random!")
        self.output_means = {
            'CB': 50.0,
            'CO2': 100.0,
            'SO2': 10.0
        }
        self.output_stds = {
            'CB': 10.0,
            'CO2': 20.0,
            'SO2': 2.0
        }
    
    def predict(self, x):
        """Return random predictions within reasonable ranges"""
        return {
            'CB': np.random.normal(self.output_means['CB'], self.output_stds['CB']),
            'CO2': np.random.normal(self.output_means['CO2'], self.output_stds['CO2']),
            'SO2': np.random.normal(self.output_means['SO2'], self.output_stds['SO2'])
        }

def load_lstm_model(model_path=None):
    """Load LSTM model with proper error handling"""
    try:
        from tensorflow.keras.models import load_model
        
        if model_path is None:
            model_path = 'models/lstm_model.h5'
        
        model = load_model(model_path)
        logging.info(f"Successfully loaded LSTM model from {model_path}")
        return model
    
    except Exception as e:
        logging.warning(f"Failed to load LSTM model: {str(e)}")
        return DummyPredictor() 