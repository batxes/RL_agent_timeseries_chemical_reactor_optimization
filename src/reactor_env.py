import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tensorflow.keras.models import load_model
import logging
import os
from datetime import datetime

class ChemicalReactorEnv(gym.Env):
    def __init__(self, reactor_number, weights=None, active_parameters=None):
        """
        Initialize environment with multiple LSTM models for different objectives.
        
        Args:
            reactor_number: Reactor number (e.g., 2)
            weights: Dictionary with weights for each objective
                    Default: CB=0.5 (production), CO2=0.25, SO2=0.25 (emissions)
        """
        super().__init__()
        self.reactor_number = reactor_number

        # Set default weights if none provided
        self.weights = weights if weights is not None else {
            'CB': 0.5,    # Higher weight for production
            'CO2': 0.25,  # Lower weights for emissions
            'SO2': 0.25
        }

        # Load models first to get input shape
        try:
            self.cb_model = load_model(f'models/lstm_model_reactor_{reactor_number}_target_{reactor_number}|CB.keras')
            self.co2_model = load_model(f'models/lstm_model_reactor_{reactor_number}_target_R{reactor_number} CO2.keras')
            self.so2_model = load_model(f'models/lstm_model_reactor_{reactor_number}_target_R{reactor_number} SO2.keras')
            
            # Get expected input shape from model
            self.expected_timesteps = self.cb_model.input_shape[1]
            self.expected_features = self.cb_model.input_shape[2]
            logging.info(f"Model expects input shape: (batch, {self.expected_timesteps}, {self.expected_features})")
            
            self.using_dummy_model = False
        except Exception as e:
            logging.warning(f"Error loading models: {e}. Using dummy predictor.")
            self.using_dummy_model = True
            self.expected_timesteps = 5
            self.expected_features = 47

        # Define controllable parameters and their ranges
        self.full_parameter_config = {
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

                # Filter parameters based on selection
        if active_parameters:
            self.parameter_config = {
                param: self.full_parameter_config[param]
                for param in active_parameters
                if param in self.full_parameter_config
            }
        else:
            self.parameter_config = self.full_parameter_config
            
        self.parameter_names = list(self.parameter_config.keys())
        # Update parameter_indices to only track active parameters
        self.parameter_indices = {
            param: i for i, param in enumerate(self.parameter_names)
        }

        # Store full parameter list for state management
        self.full_parameter_list = list(self.full_parameter_config.keys())
        self.full_state = np.array([
            (self.full_parameter_config[param][0] + self.full_parameter_config[param][1]) / 2
            for param in self.full_parameter_list
        ])
        
        # Define action and observation spaces only for selected parameters
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

        

        # Define action and observation spaces (keep your existing spaces)
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
        # Initialize full state with middle values
        self.full_state = np.array([
            (self.full_parameter_config[param][0] + self.full_parameter_config[param][1]) / 2
            for param in self.full_parameter_list
        ])
        
        # Return only active parameters
        self.state = np.array([
            self.full_state[self.full_parameter_list.index(param)]
            for param in self.parameter_names
        ])
        return self.state, {}

    def _calculate_reward(self, predictions):
        """Calculate weighted reward from all objectives"""
        if not hasattr(self, 'last_predictions'):
            self.last_predictions = predictions
            return 0
        
        # Calculate improvements (or reductions) for each objective
        improvements = {
            'CB': (predictions['CB'] - self.last_predictions['CB']) / max(1e-6, self.last_predictions['CB']),
            'CO2': -(predictions['CO2'] - self.last_predictions['CO2']) / max(1e-6, self.last_predictions['CO2']),
            'SO2': -(predictions['SO2'] - self.last_predictions['SO2']) / max(1e-6, self.last_predictions['SO2'])
        }
        
        # Store current predictions for next step
        self.last_predictions = predictions
        
        # Calculate weighted sum of improvements
        reward = sum(self.weights[obj] * imp for obj, imp in improvements.items())
        
        return reward
    
    def step(self, action):
        """
        Take a step in the environment using only active parameters.
        """
        # Clip action to valid ranges
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update full state with new actions for active parameters
        for i, param in enumerate(self.parameter_names):
            full_idx = self.full_parameter_list.index(param)
            self.full_state[full_idx] = clipped_action[i]
        
        # Update current state with active parameters
        self.state = np.array([
            self.full_state[self.full_parameter_list.index(param)]
            for param in self.parameter_names
        ])
        
        # Get predictions using full state
        predictions = self._predict_outputs(self.full_state)
        
        # Calculate reward
        reward = self._calculate_reward(predictions)
        
        # Add penalties for large parameter changes
        penalty = self._calculate_penalties(clipped_action)
        reward -= penalty
        
        # Episode never done (continuous optimization)
        done = False
        
        # Include predictions in info
        info = {
            'CB_pred': predictions['CB'],
            'CO2_pred': predictions['CO2'],
            'SO2_pred': predictions['SO2'],
            'reward': reward,
            'penalty': penalty
        }
        
        return self.state, reward, done, False, info

    def _prepare_model_input(self, state):
        """
        Prepare the input for LSTM models.
        Expected shape: (batch_size, timesteps, features)
        """
        try:
            # Define expected dimensions
            n_timesteps = 5
            n_features = 47  # Total number of features expected by the model
            
            # Create a full feature vector with zeros
            full_features = np.zeros(n_features)
            
            # Map the active parameters to their correct positions in the full feature vector
            for param, value in zip(self.parameter_names, state):
                if param in self.full_parameter_config:
                    # Get the index in the full feature set
                    # You might need to adjust this mapping based on your actual feature order
                    feature_idx = list(self.full_parameter_config.keys()).index(param)
                    full_features[feature_idx] = value
            
            # Repeat the features for the required timesteps
            # Shape: (1, timesteps, features)
            model_input = np.repeat(full_features[np.newaxis, np.newaxis, :], 
                                n_timesteps, 
                                axis=1)
            
            logging.debug(f"Prepared model input shape: {model_input.shape}")
            return model_input
            
        except Exception as e:
            logging.error(f"Error preparing model input: {e}")
            raise

    def _prepare_model_input_old(self, state):
        """
        Prepare the input for LSTM models.
        Expected shape: (batch_size, timesteps, features)
        """        
        if self.using_dummy_model:
            return np.array([state])
        
        try:
            # Create a dictionary with current parameter values
            current_values = {name: value for name, value in zip(self.parameter_names, state)}
            
            # Get current time features
            now = datetime.now()
            current_hour = now.hour
            current_day = now.day
            current_month = now.month
            current_weekday = now.weekday()
            is_weekend = 1 if current_weekday >= 5 else 0
            
            # Create full feature vector including rolling and lag features
            # For now, we'll use dummy values for historical features
            feature_vector = []
            
            # Add controllable parameters first
            for param in self.parameter_names:
                feature_vector.append(current_values[param])
            
            # Add rolling features (using dummy values for now)
            target = f"{self.reactor_number}|CB"
            for window in [3, 6, 12, 24]:
                feature_vector.extend([50.0] * 4)  # mean, std, min, max
            
            # Add lag features (using dummy values for now)
            for lag in [1, 2, 3, 6, 12, 24]:
                feature_vector.append(50.0)
            
            # Add time features
            feature_vector.extend([
                current_hour,
                current_day,
                current_month,
                current_weekday,
                is_weekend
            ])
                  # Convert to numpy array and reshape for LSTM
            feature_vector = np.array(feature_vector)
            
            # Create sequence of 5 timesteps
            sequence = np.tile(feature_vector, (5, 1))  # Shape: (5, n_features)
            
            # Add batch dimension
            model_input = np.expand_dims(sequence, axis=0)  # Shape: (1, 5, n_features)
            
            # Validate shape
            expected_shape = (1, 5, len(feature_vector))
            if model_input.shape != expected_shape:
                raise ValueError(f"Invalid input shape. Expected {expected_shape}, got {model_input.shape}")
            
            logging.debug(f"Prepared input shape: {model_input.shape}")
            return model_input
            
        except Exception as e:
            logging.error(f"Error preparing model input: {e}")
            self.using_dummy_model = True
            return self._prepare_model_input(state)

    def _predict_outputs(self, state):
        """Predict all objectives using respective models"""
        if self.using_dummy_model:
            return {
                'CB': np.mean(state) * 1.5,  # Dummy production
                'CO2': 100 - np.mean(state),  # Dummy CO2 emissions
                'SO2': 50 - np.mean(state)    # Dummy SO2 emissions
            }
        
        try:
            model_input = self._prepare_model_input(state)
            return {
                'CB': float(self.cb_model.predict(model_input, verbose=0)[0][0]),
                'CO2': float(self.co2_model.predict(model_input, verbose=0)[0][0]),
                'SO2': float(self.so2_model.predict(model_input, verbose=0)[0][0])
            }
        except Exception as e:
            logging.warning(f"Error in prediction: {e}. Using dummy values.")
            self.using_dummy_model = True
            return self._predict_outputs(state)

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