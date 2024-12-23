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


class ReactorOptimizationAgent:
    def __init__(self, reactor_number, weights=None, active_parameters=None):
        """Initialize the agent with a chemical reactor environment."""
        try:
            # Create the raw environment
            self.raw_env = ChemicalReactorEnv(
                reactor_number=reactor_number,
                weights=weights,
                active_parameters=active_parameters
            )
            logging.info("Raw environment created successfully in agent")
            
            # Calculate network size based on number of parameters
            n_parameters = len(active_parameters) if active_parameters else len(self.raw_env.parameter_names)
            logging.info(f"Number of parameters to optimize: {n_parameters}")
            
            # Scale network architecture based on parameter count
            if n_parameters <= 2:
                net_arch = dict(pi=[16, 8], vf=[16, 8])  # Tiny network for few parameters
                batch_size = 32
                learning_rate = 5e-4  # Increased for faster learning
                n_steps = 256  # Reduced for faster updates
                n_epochs = 4   # Fewer training epochs
                exploration_noise = -2.5  # More focused exploration
            elif n_parameters <= 5:
                net_arch = dict(pi=[32, 16], vf=[32, 16])  # Small network
                batch_size = 64
                learning_rate = 2e-4
                n_steps = 512
                n_epochs = 6
                exploration_noise = -2.0
            elif n_parameters <= 10:
                net_arch = dict(pi=[64, 32], vf=[64, 32])  # Medium network
                batch_size = 128
                learning_rate = 1e-4
                n_steps = 1024
                n_epochs = 8
                exploration_noise = -1.5
            else:
                net_arch = dict(pi=[128, 64], vf=[128, 64])  # Large network
                batch_size = 256
                learning_rate = 5e-5
                n_steps = 2048
                n_epochs = 10
                exploration_noise = -1.0
            
            # Initialize training metrics
            self.training_metrics = {
                'steps_per_second': [],
                'mean_reward': [],
                'timesteps': [],
                'parameter_updates': {},
                'network_size': sum([np.prod(layer) for layer in net_arch['pi'] + net_arch['vf']])
            }
            
            # Wrap environment for stable-baselines
            self.env = DummyVecEnv([lambda: self.raw_env])
            
            # Create PPO model with scaled architecture
            self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(
                net_arch=net_arch,
                log_std_init=exploration_noise  # Adjusted exploration noise
            ),
            verbose=1
        )
            
            # Log detailed configuration
            logging.info("=" * 50)
            logging.info("PPO Configuration Details:")
            logging.info(f"Number of parameters to optimize: {n_parameters}")
            logging.info("\nNetwork Architecture:")
            logging.info(f"- Policy Network: {net_arch['pi']}")
            logging.info(f"- Value Network: {net_arch['vf']}")
            logging.info(f"- Total network parameters: {self.training_metrics['network_size']}")
            
            logging.info("\nTraining Parameters:")
            logging.info(f"- Learning rate: {learning_rate}")
            logging.info(f"- Batch size: {batch_size}")
            logging.info(f"- Steps per update (n_steps): {n_steps}")
            logging.info(f"- Training epochs: {n_epochs}")
            logging.info(f"- Exploration noise (log_std_init): {exploration_noise}")
            
            logging.info("\nPPO Specific Parameters:")
            logging.info(f"- Gamma (discount factor): {self.model.gamma}")
            logging.info(f"- GAE Lambda: {self.model.gae_lambda}")
            logging.info(f"- Clip range: {self.model.clip_range}")
            logging.info(f"- Entropy coefficient: {self.model.ent_coef}")
            logging.info(f"- Value function coefficient: {self.model.vf_coef}")
            logging.info(f"- Max gradient norm: {self.model.max_grad_norm}")
            logging.info(f"- Target kl divergence: {self.model.target_kl}")
            
            logging.info("\nEnvironment Info:")
            logging.info(f"- Observation space: {self.env.observation_space}")
            logging.info(f"- Action space: {self.env.action_space}")
            logging.info(f"- Active parameters: {active_parameters}")
            
            if weights:
                logging.info("\nOptimization Weights:")
                for key, value in weights.items():
                    logging.info(f"- {key}: {value}")
            
            logging.info("=" * 50)
            
            # Store configuration for reference
            self.config = {
                'n_parameters': n_parameters,
                'network_size': self.training_metrics['network_size'],
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'n_steps': n_steps,
                'n_epochs': n_epochs
            }
            
            # Initialize training time
            self.training_time = 0
            self.last_update_time = None
            
        except Exception as e:
            logging.error(f"Error initializing agent: {str(e)}")
            raise RuntimeError(f"Failed to initialize agent: {str(e)}")


    def configure_training(self, learning_rate, n_steps, n_epochs, clip_range):
        """Configure training parameters"""
        try:
            # Create constant schedules
            def constant_fn(val):
                return lambda _: val

            # Update model parameters
            self.model.learning_rate = constant_fn(float(learning_rate))
            self.model.n_steps = int(n_steps)
            self.model.n_epochs = int(n_epochs)
            self.model.clip_range = constant_fn(float(clip_range))
            
            # Update policy parameters
            self.model.policy.optimizer.param_groups[0]['lr'] = float(learning_rate)
            
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
            
            def _on_step(self) -> bool:
                # Update metrics every 100 steps
                if self.n_calls % 100 == 0:
                    # Get training info
                    if hasattr(self.model, 'logger') and self.model.logger is not None:
                        # Store detailed metrics
                        for key, value in self.model.logger.name_to_value.items():
                            if key not in self.agent.training_metrics:
                                self.agent.training_metrics[key] = []
                            self.agent.training_metrics[key].append(value)
                
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

    def set_parameters(self, learning_rate=None, batch_size=None, n_epochs=None):
        """Update model parameters"""
        if learning_rate is not None:
            self.model.learning_rate = learning_rate
        if batch_size is not None:
            self.model.batch_size = batch_size
        if n_epochs is not None:
            self.model.n_epochs = n_epochs

    def _create_model(self):
        """Create and configure the PPO model"""
        policy_kwargs = {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
        }
        
        # Calculate buffer size based on n_steps
        n_steps = 2048
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            n_steps=n_steps,  # Explicitly set n_steps
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            policy_kwargs=policy_kwargs,
            # Ensure buffer size is sufficient
            buffer_size=n_steps + 1  # Add extra space to prevent overflow
        )
        
        # Initialize training metrics
        self.training_metrics = {
            'timesteps': [],
            'mean_reward': [],
            'steps_per_second': [],
            'episode_rewards': [],
            'parameter_updates': {param: [] for param in self.env.parameter_names},
            'network_size': sum(p.numel() for p in self.model.policy.parameters())
        }