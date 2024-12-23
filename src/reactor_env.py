import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tensorflow.keras.models import load_model
import logging
import os
from datetime import datetime
from src.predictors import load_lstm_model, DummyPredictor
import pandas as pd

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

        # Load and analyze data to get parameter ranges
        try:
            data_path = f'data/df_cleaned_for_reactor_{reactor_number}_target_{reactor_number}|CB.tsv'
            df = pd.read_csv(data_path, sep='\t')
            
            # Define controllable parameters and their ranges based on data
            self.full_parameter_config = {}
            
            # List of parameters to analyze
            params_to_check = [
                f'{reactor_number}|Erdgas',
                f'{reactor_number}|Konst.Stufe',
                f'{reactor_number}|Perlwasser',
                f'{reactor_number}|Regelstufe',
                f'{reactor_number}|V-Luft',
                f'{reactor_number}|VL Temp',
                f'{reactor_number}|Fuelöl',
                f'{reactor_number}|Makeöl',
                f'{reactor_number}|Makeöl|Temperatur',
                f'{reactor_number}|Makeöl|Ventil',
                f'{reactor_number}|CCT',
                f'{reactor_number}|CTD',
                f'{reactor_number}|FCC',
                f'{reactor_number}|SCT',
                f'{reactor_number}|C',
                f'{reactor_number}|H',
                f'{reactor_number}|N',
                f'{reactor_number}|O',
                f'{reactor_number}|S'
            ]
            
            # Calculate ranges with safety margins
            for param in params_to_check:
                if param in df.columns:
                    min_val = df[param].min()
                    max_val = df[param].max()
                    mean_val = df[param].mean()
                    std_val = df[param].std()
                    
                    # Add 10% margin to ranges
                    margin = (max_val - min_val) * 0.1
                    safe_min = max(0, min_val - margin)  # Ensure non-negative
                    safe_max = max_val + margin
                    
                    self.full_parameter_config[param] = (safe_min, safe_max)
                    
                    # Log the ranges for verification
                    logging.info(f"Parameter {param}:")
                    logging.info(f"  Range: {safe_min:.2f} to {safe_max:.2f}")
                    logging.info(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}")
            
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
            
            # Define action and observation spaces using actual ranges
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
            
            # Log the final configuration
            logging.info("\nFinal parameter configuration:")
            for param, (low, high) in self.parameter_config.items():
                logging.info(f"{param}: [{low:.2f}, {high:.2f}]")
            
        except Exception as e:
            logging.error(f"Error setting up parameter ranges: {str(e)}")
            raise

        # Load the required models with better error handling and GPU fallback
        try:
            import tensorflow as tf
            
            # Try to disable GPU if there are CUDA issues
            try:
                tf.config.set_visible_devices([], 'GPU')
                logging.info("Disabled GPU due to potential CUDA issues. Using CPU instead.")
            except:
                logging.warning("Could not disable GPU explicitly")
            
            # Define model paths
            cb_model_path = f'models/lstm_model_reactor_{reactor_number}_target_{reactor_number}|CB.keras'
            co2_model_path = f'models/lstm_model_reactor_{reactor_number}_target_R{reactor_number} CO2.keras'
            so2_model_path = f'models/lstm_model_reactor_{reactor_number}_target_R{reactor_number} SO2.keras'
            
            logging.info("\nLooking for models:")
            logging.info(f"CB model path: {cb_model_path}")
            logging.info(f"CO2 model path: {co2_model_path}")
            logging.info(f"SO2 model path: {so2_model_path}")
            
            # Check if files exist
            missing_models = []
            for name, path in [
                ('CB', cb_model_path),
                ('CO2', co2_model_path),
                ('SO2', so2_model_path)
            ]:
                if not os.path.exists(path):
                    missing_models.append(f"{name} ({path})")
                else:
                    logging.info(f"Found {name} model at {path}")
            
            if missing_models:
                raise FileNotFoundError(
                    f"Missing LSTM models:\n" + 
                    "\n".join(missing_models)
                )
            
            # Load models with CPU fallback
            with tf.device('/CPU:0'):
                try:
                    self.cb_model = load_model(cb_model_path)
                    logging.info(f"Successfully loaded CB model from {cb_model_path}")
                    
                    self.co2_model = load_model(co2_model_path)
                    logging.info(f"Successfully loaded CO2 model from {co2_model_path}")
                    
                    self.so2_model = load_model(so2_model_path)
                    logging.info(f"Successfully loaded SO2 model from {so2_model_path}")
                    
                    # Verify models are loaded
                    if not all([self.cb_model, self.co2_model, self.so2_model]):
                        raise RuntimeError("One or more models failed to load properly")
                    
                    logging.info("All models loaded successfully!")
                    
                except Exception as e:
                    raise RuntimeError(f"Error loading models: {str(e)}")
                
        except Exception as e:
            logging.error(f"Failed to load required LSTM models: {str(e)}")
            raise RuntimeError(f"Required LSTM models not found or invalid: {str(e)}")

        # Continue with the rest of initialization
        self.reset()

    def reset(self, seed=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        try:
            # Convert dict_keys to list for all parameter configurations
            full_param_list = list(self.full_parameter_config.keys())
            active_param_list = list(self.parameter_config.keys())
            
            # Initialize full state with middle values
            self.full_state = np.zeros(len(full_param_list))
            for i, param in enumerate(full_param_list):
                min_val, max_val = self.full_parameter_config[param]
                self.full_state[i] = (min_val + max_val) / 2
                
            # Return only active parameters
            self.state = np.array([
                self.full_state[full_param_list.index(param)]
                for param in active_param_list
            ])
            
            logging.debug("Reset state:")
            logging.debug(f"Full parameters: {full_param_list}")
            logging.debug(f"Active parameters: {active_param_list}")
            logging.debug(f"State shape: {self.state.shape}")
            
            return self.state, {}
            
        except Exception as e:
            logging.error(f"Error in reset: {str(e)}")
            logging.error("Parameter configurations:")
            logging.error(f"Full config: {self.full_parameter_config}")
            logging.error(f"Active config: {self.parameter_config}")
            raise

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
        """Take a step in the environment using only active parameters."""
        try:
            # Convert dict_keys to list for indexing
            full_param_list = list(self.full_parameter_config.keys())
            active_param_list = list(self.parameter_config.keys())
            
            # Clip action to valid ranges
            clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
            
            # Update full state with new actions for active parameters
            for i, param in enumerate(active_param_list):
                full_idx = full_param_list.index(param)
                self.full_state[full_idx] = clipped_action[i]
            
            # Update current state with active parameters
            self.state = np.array([
                self.full_state[full_param_list.index(param)]
                for param in active_param_list
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
            
        except Exception as e:
            logging.error(f"Error in step: {str(e)}")
            logging.error(f"Action shape: {action.shape if hasattr(action, 'shape') else type(action)}")
            logging.error(f"Current state shape: {self.state.shape if hasattr(self.state, 'shape') else type(self.state)}")
            raise

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
        """Predict outputs using LSTM models"""
        try:
            # Create a full feature vector with zeros (47 features)
            full_features = np.zeros(47)
            
            # Convert dict_keys to list for indexing
            full_parameter_list = list(self.full_parameter_config.keys())
            
            # Map the active parameters to their correct positions
            for i, param in enumerate(self.parameter_names):
                try:
                    idx = full_parameter_list.index(param)
                    full_features[idx] = state[i] if isinstance(state, np.ndarray) else state
                except ValueError:
                    logging.error(f"Parameter {param} not found in full parameter list")
                    logging.error(f"Available parameters: {full_parameter_list}")
                    raise
                
            # Log the mapping for debugging
            logging.debug("Parameter mapping:")
            for i, param in enumerate(self.parameter_names):
                idx = full_parameter_list.index(param)
                logging.debug(f"  {param} -> position {idx}: value {full_features[idx]}")
            
            # Reshape for LSTM input (batch_size, timesteps=5, features=47)
            state_array = np.tile(full_features, (5, 1))  # Repeat features for 5 timesteps
            state_array = np.expand_dims(state_array, axis=0)  # Add batch dimension
            
            # Verify shape before prediction
            if state_array.shape != (1, 5, 47):
                raise ValueError(f"Invalid input shape: expected (1, 5, 47), got {state_array.shape}")
            
            # Get predictions using model.__call__ instead of predict
            predictions = {}
            for model_name, model in [('CB', self.cb_model), ('CO2', self.co2_model), ('SO2', self.so2_model)]:
                try:
                    prediction = model(state_array, training=False)
                    predictions[model_name] = float(prediction.numpy()[0, 0])
                except Exception as e:
                    logging.error(f"Error predicting {model_name}: {str(e)}")
                    predictions[model_name] = 0.0
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error in _predict_outputs: {str(e)}")
            logging.error(f"State shape: {np.array(state).shape if hasattr(state, 'shape') else type(state)}")
            # Return default predictions instead of raising
            return {'CB': 0.0, 'CO2': 0.0, 'SO2': 0.0}

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