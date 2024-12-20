import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tensorflow.keras.models import load_model
import logging
import os

class ChemicalReactorEnv(gym.Env):
    def __init__(self, reactor_number, optimization_target="CB"):
        """
        optimization_target: "CB" (production), "CO2" (emissions), or "SO2" (emissions)
        """
        super().__init__()
        self.reactor_number = reactor_number
        self.optimization_target = optimization_target

        # Try to load model, use dummy predictor if not available
        model_target = f"{reactor_number}|CB" if optimization_target == "CB" else f"R{reactor_number} {optimization_target}"
        model_path = f'models/lstm_model_reactor_{reactor_number}_target_{model_target}.keras'
        
        try:
            if os.path.exists(model_path):
                self.prediction_model = load_model(model_path)
                self.using_dummy_model = False
                logging.info(f"Loaded model from {model_path}")
            else:
                logging.warning(f"Model not found at {model_path}. Using dummy predictor.")
                self.using_dummy_model = True
        except Exception as e:
            logging.warning(f"Error loading model: {e}. Using dummy predictor.")
            self.using_dummy_model = True
        
        # Define controllable parameters and their ranges
        self.parameter_config = {
            f'{reactor_number}|Erdgas': (0, 100),
            f'{reactor_number}|Konst.Stufe': (0, 100),
            f'{reactor_number}|Perlwasser': (0, 100),
            f'{reactor_number}|Regelstufe': (0, 100),
            f'{reactor_number}|Sorte': (0, 100),
            f'{reactor_number}|V-Luft': (0, 100),
            f'{reactor_number}|VL Temp': (0, 300),
            f'{reactor_number}|Fuelöl': (0, 100),
            f'{reactor_number}|Makeöl': (0, 100),
            f'{reactor_number}|Makeöl|Temperatur': (0, 300),
            f'{reactor_number}|Makeöl|Ventil': (0, 100),
            f'{reactor_number}|CCT': (0, 100),
            f'{reactor_number}|CTD': (0, 100),
            f'{reactor_number}|FCC': (0, 100),
            f'{reactor_number}|SCT': (0, 100),
            f'{reactor_number}|C': (0, 100),
            f'{reactor_number}|H': (0, 100),
            f'{reactor_number}|N': (0, 100),
            f'{reactor_number}|O': (0, 100),
            f'{reactor_number}|S': (0, 100)
        }
        
        # Define action and observation spaces
        self.parameter_names = list(self.parameter_config.keys())
        self.action_space = spaces.Box(
            low=np.array([config[0] for config in self.parameter_config.values()]),
            high=np.array([config[1] for config in self.parameter_config.values()]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.action_space.low,
            high=self.action_space.high,
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize state with middle values
        self.state = np.array([
            (high + low) / 2 for low, high in self.parameter_config.values()
        ])
        return self.state, {}

    def step(self, action):
        # Apply action (new parameter values)
        self.state = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Predict output using LSTM model
        output = self._predict_output(self.state)
        
        # Calculate reward based on optimization target
        if self.optimization_target == "CB":
            # For production, we want to maximize
            reward = output - self.last_output if hasattr(self, 'last_output') else 0
        else:
            # For emissions (CO2, SO2), we want to minimize
            reward = self.last_output - output if hasattr(self, 'last_output') else 0
        
        self.last_output = output
        
        # Add penalties
        penalty = self._calculate_penalties(action)
        reward -= penalty
        
        # Episode never done (continuous optimization)
        done = False
        
        return self.state, reward, done, False, {'output': output}

    def _prepare_model_input(self, state):
        """Prepare state for model input"""
        if self.using_dummy_model:
            return np.array([state])
        
        try:
            # Convert state to dictionary format
            state_dict = {name: value for name, value in zip(self.parameter_names, state)}
            
            # Create sequence of 5 timesteps (as per your training setup)
            sequence = np.array([list(state_dict.values())] * 5)  # Repeat current state 5 times
            
            # Reshape for LSTM input: [batch_size, time_steps, features]
            model_input = np.expand_dims(sequence, axis=0)  # Shape: (1, 5, n_features)
            
            logging.debug(f"Model input shape: {model_input.shape}")
            return model_input
            
        except Exception as e:
            logging.error(f"Error preparing model input: {e}")
            self.using_dummy_model = True
            return self._prepare_model_input(state)  # Recursive call will use dummy logic

    def _predict_output(self, state):
        """Predict output based on state"""
        if self.using_dummy_model:
            # Dummy prediction logic
            if self.optimization_target == "CB":
                # For production, return a value based on average of parameters
                return np.mean(state) * 1.5
            else:
                # For emissions, return a value inversely proportional to parameters
                return 100 - np.mean(state)
        else:
            try:
                # Use actual model for prediction
                model_input = self._prepare_model_input(state)
                
                # Add input shape validation
                expected_shape = (1, 5, len(self.parameter_names))  # batch_size, time_steps, features
                if model_input.shape != expected_shape:
                    raise ValueError(
                        f"Invalid input shape. Expected {expected_shape}, got {model_input.shape}"
                    )
                
                prediction = self.prediction_model.predict(model_input, verbose=0)
                return float(prediction[0][0])
                
            except Exception as e:
                logging.warning(f"Error in prediction: {e}. Falling back to dummy predictor.")
                self.using_dummy_model = True
                return self._predict_output(state)  # Recursive call will use dummy logic

    def _calculate_penalties(self, action):
        penalty = 0
        
        # Penalty for extreme values
        for value, (low, high) in zip(action, self.parameter_config.values()):
            range_size = high - low
            margin = range_size * 0.1  # 10% margin from extremes
            if value < low + margin or value > high - margin:
                penalty += 0.1
        
        # Penalty for rapid changes
        changes = np.abs(action - self.state)
        max_allowed_changes = (self.action_space.high - self.action_space.low) * 0.1  # 10% max change
        penalty += np.sum(np.maximum(0, changes - max_allowed_changes)) * 0.01
        
        return penalty