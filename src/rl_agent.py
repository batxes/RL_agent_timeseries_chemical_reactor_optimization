from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from src.reactor_env import ChemicalReactorEnv
import logging

class ReactorOptimizationAgent:
    def __init__(self, reactor_number, optimization_target="CB"):
        """
        optimization_target: "CB" (production), "CO2" (emissions), or "SO2" (emissions)
        """
        self.reactor_number = reactor_number
        self.optimization_target = optimization_target
        
        try:
        # Create and wrap the environment
            env = ChemicalReactorEnv(reactor_number, optimization_target)
            self.env = DummyVecEnv([lambda: env])
            
            # Initialize PPO agent with custom parameters
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=[256, 256],  # Policy network
                        vf=[256, 256]   # Value function network
                    )
                ),
                verbose=1
            )
        except Exception as e:
            logging.error(f"Error initializing agent: {e}")
            raise

    def train(self, total_timesteps=50000):
        self.model.learn(total_timesteps=total_timesteps)
        
    def save(self, path):
        self.model.save(path)
        
    def load(self, path):
        self.model = PPO.load(path, env=self.env)
        
    def suggest_parameters(self, current_state):
        state_array = np.array([current_state[param] for param in self.env.get_attr('parameter_names')[0]])
        action, _ = self.model.predict(state_array, deterministic=True)
        
        suggestions = {}
        for param, value in zip(self.env.get_attr('parameter_names')[0], action):
            suggestions[param] = float(value)
            
        return suggestions