from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from src.reactor_env import ChemicalReactorEnv
import logging
import torch
from tensorflow.keras.models import load_model
import time
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

class TrainingCallback(BaseCallback):
    def __init__(self, dashboard, verbose=0):
        super().__init__(verbose)
        self.dashboard = dashboard
        
    def _on_step(self):
        # Update dashboard metrics
        if self.n_calls % 1000 == 0:  # Update every 1000 steps
            self.dashboard['metrics_history'].append({
                'timesteps': self.num_timesteps,
                'reward': self.training_env.buf_rews[0],
                'episode_length': self.n_calls
            })
        return True

class MultiObjectiveReactorEnv(ChemicalReactorEnv):

    def __init__(self, reactor_number, weights=None):
        super().__init__(reactor_number, weights)
        
    def step(self, action):
        """
        Take a step in the environment.
        Args:
            action: numpy array of parameter values
        Returns:
            (observation, reward, done, truncated, info)
        """
        # Apply action (new parameter values)
        self.state = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Get predictions for all objectives
        predictions = self._predict_outputs(self.state)
        
        # Calculate multi-objective reward
        reward = self._calculate_reward(predictions)
        
        # Add penalties for large parameter changes
        penalty = self._calculate_penalties(action)
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

    def _calculate_penalties(self, action):
        """Calculate penalties for large parameter changes"""
        # Calculate relative changes
        changes = np.abs(action - self.state) / (self.action_space.high - self.action_space.low)
        
        # Penalty increases quadratically with change size
        penalty = np.mean(changes ** 2) * 0.1
        
        return penalty

class ReactorOptimizationAgent:
    def __init__(self, reactor_number, weights=None, active_parameters=None):
        self.reactor_number = reactor_number
        self.weights = weights
        
        # Create environment with selected parameters
        self.raw_env = ChemicalReactorEnv(
            reactor_number=reactor_number,
            weights=weights,
            active_parameters=active_parameters
        )
        
        # Initialize training metrics
        self.training_metrics = {
            'steps_per_second': [],
            'mean_reward': [],
            'parameter_updates': {},
            'timesteps': []
        }
        
        # Calculate complexity metrics
        self.complexity_metrics = {
            'action_space_size': len(active_parameters) if active_parameters else len(self.raw_env.parameter_names),
            'total_combinations': np.prod([
                self.raw_env.parameter_config[param][1] - self.raw_env.parameter_config[param][0]
                for param in self.raw_env.parameter_names
            ])
        }
        
        # Initialize parameter tracking
        for param in self.raw_env.parameter_names:
            self.training_metrics['parameter_updates'][param] = []
        
        # Wrap environment for stable-baselines
        self.env = DummyVecEnv([lambda: self.raw_env])
        
        # Create PPO model with smaller network for fewer parameters
        policy_kwargs = dict(
            net_arch=dict(
                pi=[64, 32],  # Smaller policy network
                vf=[64, 32]   # Smaller value network
            )
        )
        
        # Adjust batch size based on number of parameters
        batch_size = min(64, max(32, self.complexity_metrics['action_space_size'] * 4))
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1
        )
        
        # Initialize training time
        self.training_time = 0
        self.last_update_time = None


    def configure_training(self, learning_rate, n_steps, n_epochs, clip_range):
        """
        Configure training parameters.
        
        Args:
            learning_rate: Learning rate for the optimizer
            n_steps: Number of steps to run for each environment per update
            n_epochs: Number of epochs when optimizing the surrogate loss
            clip_range: Clipping parameter for PPO
        """
        try:
            def constant_fn(val):
                return lambda _: val

            # Create a new model with updated parameters
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=constant_fn(float(learning_rate)),
                n_steps=int(n_steps),
                batch_size=128,
                n_epochs=int(n_epochs),
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=constant_fn(float(clip_range)),
                clip_range_vf=constant_fn(float(clip_range)),
                normalize_advantage=True,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=[128, 128],
                        vf=[128, 128]
                    ),
                    activation_fn=torch.nn.Tanh
                ),
                verbose=1,
                device='cpu'
            )
        except Exception as e:
            logging.error(f"Error configuring training parameters: {e}")
            raise

    def train(self, total_timesteps, callback=None):
        """Train the agent with performance monitoring"""
        training_start = time.time()
        self.last_update_time = training_start
        
        class MetricsCallback(BaseCallback):
            def __init__(self, agent, verbose=0):
                super().__init__(verbose)
                self.agent = agent
                self.last_update_time = time.time()
            
            def _on_step(self) -> bool:
                current_time = time.time()
                
                # Update metrics every 1000 steps
                if self.n_calls % 1000 == 0:
                    # Calculate steps per second
                    steps_per_second = 1000 / (current_time - self.last_update_time)
                    self.agent.training_metrics['steps_per_second'].append(steps_per_second)
                    self.agent.training_metrics['timesteps'].append(self.n_calls)
                    
                    # Track parameter changes if possible
                    if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
                        if self.model.rollout_buffer.observations.size > 0:
                            for i, param in enumerate(self.agent.raw_env.parameter_names):
                                values = self.model.rollout_buffer.observations[:, i]
                                if param not in self.agent.training_metrics['parameter_updates']:
                                    self.agent.training_metrics['parameter_updates'][param] = []
                                self.agent.training_metrics['parameter_updates'][param].append(
                                    np.std(values)
                                )
                    
                    self.last_update_time = current_time
                
                return True
        
        # Create callbacks list
        callbacks = []
        if callback is not None:
            callbacks.append(callback)
        callbacks.append(MetricsCallback(self))
        
        # Create callback list
        callback_list = CallbackList(callbacks)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list
        )
        
        # Store total training time
        self.training_time = time.time() - training_start
        
    def save(self, path):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model to
        """
        try:
            self.model.save(path)
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise
    
    def load(self, path):
        """
        Load a trained model.
        
        Args:
            path: Path to load the model from
        """
        try:
            self.model = PPO.load(path, env=self.env)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def suggest_parameters(self, current_state):
        """Get parameter suggestions from the trained agent"""
        # Convert current state to array
        state_array = np.array([
            current_state[param] 
            for param in self.raw_env.parameter_names
        ])
        
        # Get action from model
        action, _ = self.model.predict(state_array, deterministic=True)
        
        # Convert action back to dictionary
        suggestions = {}
        for i, param in enumerate(self.raw_env.parameter_names):
            suggestions[param] = float(action[i])
        
        return suggestions