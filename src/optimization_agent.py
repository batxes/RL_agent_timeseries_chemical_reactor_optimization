from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class OptimizationAgent:
    def __init__(self, reactor_number):
        self.reactor_number = reactor_number
        self.model = load_model(f'models/lstm_model_reactor_{reactor_number}_target_{reactor_number}|CB.keras')
        # Load scaler parameters
        # You'll need to save these during training
        self.scaler = MinMaxScaler()
        
    def suggest_parameters(self, current_state, target_increase=0.1):
        """Suggest parameter changes to improve production"""
        # Convert current state to model input format
        X = self._prepare_input(current_state)
        
        # Get current prediction
        current_prediction = self.model.predict(X)
        
        # Simple gradient-based optimization
        suggestions = {}
        for param, value in current_state.items():
            # Test small changes in each parameter
            delta = value * 0.1
            test_state = current_state.copy()
            test_state[param] = value + delta
            
            X_test = self._prepare_input(test_state)
            new_prediction = self.model.predict(X_test)
            
            if new_prediction > current_prediction:
                suggestions[param] = value + delta
        
        return suggestions
    
    def _prepare_input(self, state):
        """Prepare state dictionary for model input"""
        # Convert state to correct format for your model
        # This will depend on your model's expected input format
        return np.array([state])  # Placeholder