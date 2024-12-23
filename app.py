import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.rl_agent import ReactorOptimizationAgent
from src.reactor_env import ChemicalReactorEnv
import logging
from stable_baselines3.common.callbacks import BaseCallback
import os
from matplotlib.gridspec import GridSpec
import time
from datetime import timedelta
import traceback

def create_training_controls(agent, active_parameters):
    """Create training controls in the sidebar"""
    with st.sidebar:
        st.subheader("Training Parameters")
        
        # Model information
        n_params = len(active_parameters) if active_parameters else "all"
        st.info(f"Model Configuration:\n"
                f"- Parameters to optimize: {n_params}\n"
                f"- Network complexity: {agent.training_metrics['network_size']:,} parameters\n"
                f"- Batch size: {agent.model.batch_size}")
        
        # Training parameters
        params = {}
        params['total_steps'] = st.number_input("Training Steps", min_value=1000, value=10000, step=1000)
        params['learning_rate'] = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, 
                                                value=1e-4, format="%.0e")
        params['batch_size'] = st.number_input("Batch Size", min_value=32, max_value=512, value=64, step=32)
        params['n_epochs'] = st.number_input("Number of Epochs", min_value=1, max_value=20, value=10)
        
        # Training button
        train_button = st.button("Train Agent")
        
        # Speed metrics containers
        speed_metrics = {
            'steps_per_second': st.empty(),
            'time_per_step': st.empty(),
            'estimated_time': st.empty(),
            'active_params': st.empty()
        }
        
        # Progress information
        progress_container = st.empty()
        progress_bar = st.progress(0.0)
        
        return params, train_button, progress_container, progress_bar, speed_metrics

def create_metric_charts():
    """Create placeholder charts in the sidebar for real-time updates"""
    with st.sidebar:
        st.subheader("Training Metrics")
        
        # Create metrics containers
        metrics = {}
        
        # Training Progress
        metrics['progress_text'] = st.empty()  # For showing progress percentage
        
        # Charts
        st.markdown("### Episode Reward")
        metrics['reward_chart'] = st.empty()  # For reward plot
        st.markdown("- Green area: > 0.7 (Good)")
        st.markdown("- Red area: < 0.3 (Poor)")
        
        st.markdown("### Policy Loss")
        metrics['loss_chart'] = st.empty()  # For policy loss plot
        st.markdown("- Green area: < 0.02 (Good)")
        st.markdown("- Red area: > 0.1 (Poor)")
        
        st.markdown("### Value Loss")
        metrics['value_chart'] = st.empty()  # For value loss plot
        st.markdown("- Green area: < 0.1 (Good)")
        st.markdown("- Red area: > 0.5 (Poor)")
        
        st.markdown("### Action Standard Deviation")
        metrics['std_chart'] = st.empty()  # For std plot
        st.markdown("- Green area: 0.1 - 0.3 (Good)")
        st.markdown("- Red area: < 0.1 or > 0.3 (Poor)")
        
        return metrics

class TrainingCallback(BaseCallback):
    def __init__(self, total_steps, metric_charts, progress_container, progress_bar, speed_metrics, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.metric_charts = metric_charts
        self.progress_container = progress_container
        self.progress_bar = progress_bar
        self.speed_metrics = speed_metrics
        
        # Define thresholds for each metric
        self.thresholds = {
            'reward': {'good': 0.7, 'poor': 0.3},
            'policy_loss': {'good': 0.02, 'poor': 0.1},
            'value_loss': {'good': 0.1, 'poor': 0.5},
            'std': {'good': 0.3, 'poor': 0.1}
        }
        
        # Initialize timing variables
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_steps = 0
        
        # Initialize metric data
        self.metrics_data = {
            'rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'stds': []
        }
        
        # Create parameter display in sidebar
        with st.sidebar:
            self.param_container = st.empty()
    
    def _plot_metric_with_thresholds(self, data, metric_name, chart_container):
        """Plot metric with threshold lines and optimal areas"""
        if len(data) == 0:
            return
        
        # Create DataFrame with metric data
        df = pd.DataFrame()
        df['value'] = data
        df.index = range(len(data))
        
        # Add threshold lines
        good_threshold = self.thresholds[metric_name]['good']
        poor_threshold = self.thresholds[metric_name]['poor']
        
        # Create the optimal area based on metric type
        if metric_name in ['policy_loss', 'value_loss']:
            # For losses, lower is better
            df['optimal_area'] = good_threshold
            df['warning_area'] = poor_threshold
        elif metric_name == 'std':
            # For std, medium range is better
            df['optimal_min'] = poor_threshold
            df['optimal_max'] = good_threshold
        else:  # reward
            # For reward, higher is better
            df['optimal_area'] = good_threshold
            df['warning_area'] = poor_threshold
        
        # Plot with Streamlit
        chart_container.line_chart(df, use_container_width=True)
    
    def _on_step(self) -> bool:
        """Update progress and metrics in real-time"""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        # Update progress and parameters every 10 steps
        if self.num_timesteps % 10 == 0:
            # Get current parameters and their values
            if hasattr(self.training_env, 'get_attr'):
                param_names = self.training_env.get_attr('parameter_names')[0]
                current_state = self.training_env.get_attr('state')[0]
                
                # Create parameter table
                param_data = []
                for name, value in zip(param_names, current_state):
                    param_data.append({
                        "Parameter": name,
                        "Value": f"{value:.2f}"
                    })
                
                # Display parameters in sidebar
                self.param_container.table(pd.DataFrame(param_data))
            
            # Get metrics directly from the model's rollout buffer
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
                # Get reward
                if hasattr(self.model.rollout_buffer, 'rewards'):
                    mean_reward = np.mean(self.model.rollout_buffer.rewards)
                    self.metrics_data['rewards'].append(mean_reward)
                    logging.info(f"Updating reward chart with value: {mean_reward}")
                    self._plot_metric_with_thresholds(
                        self.metrics_data['rewards'],
                        'reward',
                        self.metric_charts['reward_chart']
                    )
                else:
                    logging.warning("No rewards found in rollout buffer")
                
                # Get action std from the policy
                if hasattr(self.model.policy, 'log_std'):
                    action_std = np.exp(self.model.policy.log_std.detach().numpy().mean())
                    self.metrics_data['stds'].append(action_std)
                    logging.info(f"Updating std chart with value: {action_std}")
                    self._plot_metric_with_thresholds(
                        self.metrics_data['stds'],
                        'std',
                        self.metric_charts['std_chart']
                    )
                else:
                    logging.warning("No log_std found in policy")
                
                # Update policy loss data
                if 'train/policy_loss' in self.model.logger.name_to_value:
                    self.metrics_data['policy_losses'].append(self.model.logger.name_to_value['train/policy_loss'])
                    self._plot_metric_with_thresholds(
                        self.metrics_data['policy_losses'], 
                        'policy_loss', 
                        self.metric_charts['loss_chart']
                    )
                
                # Update value loss data
                if 'train/value_loss' in self.model.logger.name_to_value:
                    self.metrics_data['value_losses'].append(self.model.logger.name_to_value['train/value_loss'])
                    self._plot_metric_with_thresholds(
                        self.metrics_data['value_losses'], 
                        'value_loss', 
                        self.metric_charts['value_chart']
                    )
        
        return True

def plot_training_metrics(agent):
    """Plot training metrics using matplotlib"""
    if not agent.training_metrics['timesteps']:
        return None
        
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot training speed
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(agent.training_metrics['timesteps'], 
             agent.training_metrics['steps_per_second'])
    ax1.set_title('Training Speed')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Steps/Second')
    
    # Plot mean reward
    ax2 = fig.add_subplot(gs[0, 1])
    if agent.training_metrics['mean_reward']:
        ax2.plot(agent.training_metrics['timesteps'][:len(agent.training_metrics['mean_reward'])], 
                 agent.training_metrics['mean_reward'])
        ax2.set_title('Mean Reward')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Reward')
    
    # Plot parameter evolution
    ax3 = fig.add_subplot(gs[1, 0])
    for param, values in agent.training_metrics['parameter_updates'].items():
        if values:  # Only plot if we have data
            ax3.plot(agent.training_metrics['timesteps'][:len(values)], 
                    values, label=param.split('|')[-1])
    ax3.set_title('Parameter Evolution')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Parameter Std Dev')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot reward distribution
    ax4 = fig.add_subplot(gs[1, 1])
    if agent.training_metrics['episode_rewards']:
        ax4.hist(agent.training_metrics['episode_rewards'], bins=30)
        ax4.set_title('Reward Distribution')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig

def plot_training_metrics(metrics_history):
    """Plot training metrics with good/bad ranges"""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2)
    
    # Only plot if we have data
    if not any(len(v) > 0 for v in metrics_history.values()):
        plt.figtext(0.5, 0.5, "Collecting training data...", 
                   ha='center', va='center')
        return fig
    
    # Plot 1: Learning Curves
    ax1 = fig.add_subplot(gs[0, :])
    if len(metrics_history['value_loss']) > 0:
        ax1.plot(metrics_history['value_loss'], label='Value Loss', color='blue')
    if len(metrics_history['policy_gradient_loss']) > 0:
        ax1.plot(metrics_history['policy_gradient_loss'], label='Policy Loss', color='red')
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Warning Threshold')
    ax1.fill_between(range(len(metrics_history['value_loss'])), 0, 0.5, 
                     color='green', alpha=0.1, label='Good Range')
    ax1.set_title('Learning Curves')
    ax1.set_xlabel('Updates')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot 2: Explained Variance
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(metrics_history['explained_variance'], color='purple')
    ax2.axhline(y=0.9, color='g', linestyle='--', label='Excellent', alpha=0.3)
    ax2.axhline(y=0.5, color='y', linestyle='--', label='Good', alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', label='Poor', alpha=0.3)
    ax2.fill_between(range(len(metrics_history['explained_variance'])), 0.5, 1.0,
                     color='green', alpha=0.1, label='Good Range')
    ax2.set_title('Explained Variance\n(Higher is better)')
    ax2.legend()

    # Plot 3: KL Divergence
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(metrics_history['approx_kl'], color='orange')
    ax3.axhline(y=0.2, color='r', linestyle='--', label='Max Recommended', alpha=0.3)
    ax3.fill_between(range(len(metrics_history['approx_kl'])), 0, 0.2,
                     color='green', alpha=0.1, label='Good Range')
    ax3.set_title('KL Divergence\n(Should stay below 0.2)')
    ax3.legend()

    # Plot 4: Training Speed
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(metrics_history['fps'], color='green')
    ax4.set_title('Training Speed (FPS)')
    ax4.set_xlabel('Updates')

    # Plot 5: Entropy Loss
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(metrics_history['entropy_loss'], color='red')
    ax5.set_title('Entropy Loss\n(Should decrease over time)')
    ax5.set_xlabel('Updates')

    plt.tight_layout()
    return fig

def create_training_dashboard():
    """Create a training monitoring dashboard"""
    st.subheader("Training Dashboard")
    
    # Training Parameters
    st.subheader("Training Parameters")
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.number_input(
            "Learning Rate", 
            min_value=1e-6, 
            max_value=1e-2, 
            value=3e-4, 
            format="%.0e",
            help="How quickly the agent learns (smaller = more stable but slower)"
        )
        n_steps = st.number_input(
            "Steps per Update", 
            min_value=512, 
            max_value=8192, 
            value=2048,
            help="Number of steps to run before updating policy"
        )
    
    with col2:
        n_epochs = st.number_input(
            "Number of Epochs", 
            min_value=1, 
            max_value=20, 
            value=10,
            help="How many times to reuse each batch of data"
        )
        clip_range = st.number_input(
            "Clip Range", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.2,
            help="How much the policy can change in one update"
        )
    
    # Live Metrics
    st.subheader("Training Metrics")
    metrics_cols = st.columns(4)
    metric_placeholders = {
        'reward': metrics_cols[0].empty(),
        'explained_var': metrics_cols[1].empty(),
        'policy_loss': metrics_cols[2].empty(),
        'value_loss': metrics_cols[3].empty()
    }
    
    # Training Progress
    st.subheader("Training Progress")
    progress_container = st.container()
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    # Metrics History
    metrics_history = {
        'value_loss': [], 'policy_gradient_loss': [], 
        'explained_variance': [], 'approx_kl': [],
        'fps': [], 'entropy_loss': []
    }
    
    # Plots
    plots_container = st.container()
    
    return {
        'parameters': {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'n_epochs': n_epochs,
            'clip_range': clip_range
        },
        'metric_placeholders': metric_placeholders,
        'progress_bar': progress_bar,
        'status_text': status_text,
        'metrics_history': metrics_history,
        'plots_container': plots_container
    }

def plot_model_performance(reactor_number, target):
    """Plot actual vs predicted values from test data"""
    try:
        # Load test data
        test_data = pd.read_csv(f'models/test_data_reactor_{reactor_number}_{target}.csv')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Time series plot
        ax1.plot(test_data['actual'], label='Actual', alpha=0.7)
        ax1.plot(test_data['predicted'], label='Predicted', alpha=0.7)
        ax1.set_title('Actual vs Predicted Over Time')
        ax1.legend()
        
        # Scatter plot
        ax2.scatter(test_data['actual'], test_data['predicted'], alpha=0.5)
        ax2.plot([test_data['actual'].min(), test_data['actual'].max()], 
                 [test_data['actual'].min(), test_data['actual'].max()], 
                 'r--', alpha=0.7)
        ax2.set_title('Predicted vs Actual')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        
        return fig
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None

def plot_agent_performance(env, agent, n_episodes=50):
    """Evaluate agent performance and plot results"""
    rewards = []
    improvements = []
    
    for _ in range(n_episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        initial_value = env._predict_output(state)
        best_value = initial_value
        
        while not done:
            action, _ = agent.model.predict(state, deterministic=True)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            current_value = env._predict_output(next_state)
            best_value = max(best_value, current_value)
            state = next_state
        
        rewards.append(episode_reward)
        improvements.append(((best_value - initial_value) / initial_value) * 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot reward distribution
    ax1.hist(rewards, bins=20, alpha=0.7)
    ax1.axvline(np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    ax1.set_title('Agent Reward Distribution')
    ax1.set_xlabel('Episode Reward')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # Plot improvements
    ax2.hist(improvements, bins=20, alpha=0.7)
    ax2.axvline(np.mean(improvements), color='r', linestyle='--', 
                label=f'Mean: {np.mean(improvements):.1f}%')
    ax2.set_title(f'Target Improvement Distribution')
    ax2.set_xlabel('Improvement (%)')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    return fig, np.mean(rewards), np.mean(improvements)

def analyze_training_metrics(metrics_history):
    """Analyze training metrics and provide recommendations"""
    recommendations = []
    
    # Check explained variance
    if len(metrics_history['explained_variance']) > 0:
        mean_ev = np.mean(metrics_history['explained_variance'][-10:])
        if mean_ev < 0.1:
            recommendations.append("🔴 Very low explained variance. Try:\n"
                                "- Decreasing learning rate\n"
                                "- Increasing batch size\n"
                                "- Simplifying the policy network")
    
    # Check KL divergence
    if len(metrics_history['approx_kl']) > 0:
        mean_kl = np.mean(metrics_history['approx_kl'][-10:])
        if mean_kl > 0.5:
            recommendations.append("🔴 High KL divergence. Try:\n"
                                "- Decreasing learning rate\n"
                                "- Decreasing clip range\n"
                                "- Increasing batch size")
    
    # Check value loss
    if len(metrics_history['value_loss']) > 0:
        if not np.any(np.diff(metrics_history['value_loss'][-10:]) < 0):
            recommendations.append("🟡 Value loss not decreasing. Try:\n"
                                "- Increasing training time\n"
                                "- Adjusting value function coefficient")
    
    return recommendations

def main():
    logging.basicConfig(level=logging.INFO)
    st.title("Chemical Reactor Optimization")
    
    # Initialize session state for training
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    
    # Sidebar configuration
    with st.sidebar:
        # Reactor selection with all available reactors
        reactor_number = st.selectbox(
            "Select Reactor",
            [2, 3, 4, 5, 6],  # All available reactors
            index=0,  # Default to reactor 2
            help="Select the reactor to optimize"
        )
        
        # Create a temporary environment to get parameter configurations
        try:
            env = ChemicalReactorEnv(
                reactor_number=reactor_number,
                weights={'CB': 0.5, 'CO2': 0.25, 'SO2': 0.25}  # Default weights
            )
            
            # Parameter selection using environment's configuration
            all_parameters = list(env.parameter_config.keys())
            
            active_parameters = st.multiselect(
                "Select Parameters to Optimize",
                all_parameters,
                default=[f'{reactor_number}|Erdgas']
            )
            
            # Weights configuration
            st.subheader("Optimization Weights")
            weights = {
                'CB': st.slider("Production (CB)", 0.0, 1.0, 0.5),
                'CO2': st.slider("CO2 Emissions", 0.0, 1.0, 0.25),
                'SO2': st.slider("SO2 Emissions", 0.0, 1.0, 0.25)
            }
            
        except Exception as e:
            st.error("⚠️ Cannot initialize environment: Required LSTM models not found or invalid.")
            st.error(f"Error details: {str(e)}")
            return
    
    # Initialize agent with proper error handling
    try:
        if 'agent' not in st.session_state or st.session_state.get('active_params') != active_parameters:
            agent = ReactorOptimizationAgent(
                reactor_number=reactor_number,
                weights=weights,
                active_parameters=active_parameters
            )
            st.session_state.agent = agent
            st.session_state.active_params = active_parameters
        else:
            agent = st.session_state.agent
    except Exception as e:
        st.error("⚠️ Cannot initialize training: Agent initialization failed.")
        st.error(f"Error details: {str(e)}")
        return
    
    # Create training controls and metric charts in sidebar
    params, train_button, progress_container, progress_bar, speed_metrics = create_training_controls(
        agent=st.session_state.agent,
        active_parameters=active_parameters
    )
    metric_charts = create_metric_charts()
    
    # Handle training
    if train_button:
        st.session_state.training_started = True
    
    if st.session_state.training_started:
        try:
            # Create callback with all UI elements
            callback = TrainingCallback(
                total_steps=params['total_steps'],
                metric_charts=metric_charts,
                progress_container=progress_container,
                progress_bar=progress_bar,
                speed_metrics=speed_metrics
            )
            
            # Update agent parameters
            agent.set_parameters(
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size'],
                n_epochs=params['n_epochs']
            )
            
            # Train the agent
            agent.train(total_timesteps=params['total_steps'], callback=callback)
            
            # Reset training state after completion
            st.session_state.training_started = False
            st.sidebar.success(f"Training completed in {agent.training_time:.1f} seconds!")
            
        except Exception as e:
            st.session_state.training_started = False
            st.sidebar.error(f"Training error: {str(e)}")
            logging.exception("Training error details:")

    if st.button("Show Training Metrics"):
        if hasattr(agent, 'training_metrics') and agent.training_metrics['timesteps']:
            st.write("### Training Metrics")
            
            # Show summary statistics
            st.write("**Summary Statistics:**")
            st.write(f"- Total training time: {agent.training_time:.1f} seconds")
            st.write(f"- Average steps/second: {np.mean(agent.training_metrics['steps_per_second']):.1f}")
            if agent.training_metrics['mean_reward']:
                st.write(f"- Final mean reward: {agent.training_metrics['mean_reward'][-1]:.2f}")
            
            # Show plots
            fig = plot_training_metrics(agent)
            if fig:
                st.pyplot(fig)
                
            # Show parameter importance
            st.write("\n**Parameter Activity:**")
            param_activity = {}
            for param, values in agent.training_metrics['parameter_updates'].items():
                if values:
                    param_activity[param.split('|')[-1]] = {
                        'std_dev': np.mean(values),
                        'change_rate': np.mean(np.abs(np.diff(values))) if len(values) > 1 else 0
                    }
            
            activity_df = pd.DataFrame(param_activity).T
            st.dataframe(activity_df.style.highlight_max(axis=0))
        else:
            st.warning("No training metrics available. Please train the agent first.")

    if st.button("Show Training Complexity Analysis"):
        if hasattr(agent, 'training_metrics') and agent.training_metrics['steps_per_second']:
            st.write("**Complexity Metrics:**")
            st.write(f"- Number of Active Parameters: {agent.complexity_metrics['action_space_size']}")
            st.write(f"- Total Possible Combinations: {agent.complexity_metrics['total_combinations']:.2e}")
            st.write(f"- Training Time: {agent.training_time:.2f} seconds")
            
            # Calculate efficiency metrics
            steps_per_param = params['total_steps'] / len(active_parameters)
            exploration_coverage = (agent.training_time * np.mean(agent.training_metrics['steps_per_second'])) / agent.complexity_metrics['total_combinations']
            
            st.write("\n**Efficiency Metrics:**")
            st.write(f"- Steps per Parameter: {steps_per_param:.0f}")
            st.write(f"- Search Space Coverage: {exploration_coverage:.2%}")
            st.write(f"- Average Steps/Second: {np.mean(agent.training_metrics['steps_per_second']):.1f}")
            
            # Show visualization
            st.pyplot(plot_training_complexity(agent))
        else:
            st.warning("No training metrics available. Please train the agent first.")

    # Create tabs
    tabs = st.tabs(["Model Performance", "Parameter Control", "Optimization"])
    
    # Model Performance Tab
    with tabs[0]:
        st.header("LSTM Models Performance")
        
        # Check if models exist
        model_paths = {
            'CB': f'models/lstm_model_reactor_{reactor_number}_target_{reactor_number}|CB.keras',
            'CO2': f'models/lstm_model_reactor_{reactor_number}_target_R{reactor_number} CO2.keras',
            'SO2': f'models/lstm_model_reactor_{reactor_number}_target_R{reactor_number} SO2.keras'
        }
        
        models_exist = all(os.path.exists(path) for path in model_paths.values())
        
        if models_exist:
            # Show performance for all three objectives
            for objective in ['CB', 'CO2', 'SO2']:
                st.subheader(f"{objective} Model Performance")
                fig = plot_model_performance(reactor_number, objective)
                if fig:
                    st.pyplot(fig)
                    
                    # Add performance metrics for each objective
                    test_data = pd.read_csv(f'models/test_data_reactor_{reactor_number}_{objective}.csv')
                    mse = np.mean((test_data['actual'] - test_data['predicted'])**2)
                    r2 = np.corrcoef(test_data['actual'], test_data['predicted'])[0,1]**2
                    
                    col1, col2 = st.columns(2)
                    col1.metric(f"{objective} Mean Squared Error", f"{mse:.4f}")
                    col2.metric(f"{objective} R² Score", f"{r2:.4f}")
        else:
            st.warning("Required LSTM models not found. Please ensure all models are available.")
    
    # Parameter Control Tab
    with tabs[1]:
        st.header("Parameter Control")
        
        # Group parameters by category
        parameter_groups = {
            "Main Controls": ['Erdgas', 'Konst.Stufe', 'Perlwasser', 'Regelstufe'],
            "Temperature Controls": ['VL Temp', 'Makeöl|Temperatur'],
            "Flow Controls": ['V-Luft', 'Fuelöl', 'Makeöl', 'Makeöl|Ventil'],
            "Process Parameters": ['CCT', 'CTD', 'FCC', 'SCT'],
            "Chemical Composition": ['C', 'H', 'N', 'O', 'S']
        }
        
        # Create form for parameters
        with st.form("parameter_form"):
            new_values = {}
            
            for group_name, params in parameter_groups.items():
                st.subheader(group_name)
                cols = st.columns(2)
                for i, param in enumerate(params):
                    full_param = f"{reactor_number}|{param}"
                    if full_param in env.parameter_config:
                        with cols[i % 2]:
                            min_val, max_val = env.parameter_config[full_param]
                            new_values[full_param] = st.slider(
                                param,
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=float(env.state[list(env.parameter_config.keys()).index(full_param)]))
            
            # Predict button
            if st.form_submit_button("Predict Output"):
                state_array = np.array([new_values[param] for param in env.parameter_names])
                predictions = {
                    'CB': env.cb_model.predict(state_array.reshape(1, 1, -1))[0][0],
                    'CO2': env.co2_model.predict(state_array.reshape(1, 1, -1))[0][0],
                    'SO2': env.so2_model.predict(state_array.reshape(1, 1, -1))[0][0]
                }
                
                cols = st.columns(3)
                cols[0].metric("Predicted CB", f"{predictions['CB']:.2f}")
                cols[1].metric("Predicted CO2", f"{predictions['CO2']:.2f}")
                cols[2].metric("Predicted SO2", f"{predictions['SO2']:.2f}") 
    # Optimization Tab
    with tabs[2]:
        st.header("Parameter Optimization")
        
        if st.button("Get Optimization Suggestions"):
            # Get current state and suggestions
            current_state = {param: new_values.get(param, env.state[i]) 
                           for i, param in enumerate(env.parameter_names)}
            suggestions = agent.suggest_parameters(current_state)
            
            # Display current vs suggested values
            st.subheader("Parameter Suggestions")
            
            for group_name, params in parameter_groups.items():
                st.write(f"\n**{group_name}**")
                cols = st.columns(3)
                cols[0].write("Parameter")
                cols[1].write("Current")
                cols[2].write("Suggested")
                
                for param in params:
                    full_param = f"{reactor_number}|{param}"
                    if full_param in suggestions:
                        cols = st.columns(3)
                        cols[0].write(param)
                        cols[1].write(f"{current_state[full_param]:.2f}")
                        cols[2].write(f"{suggestions[full_param]:.2f}")
            
            # Show expected improvements for all objectives
            current_state_array = np.array(list(current_state.values()))
            suggested_state_array = np.array(list(suggestions.values()))
            
            current_predictions = {
                'CB': env.cb_model.predict(current_state_array.reshape(1, 1, -1))[0][0],
                'CO2': env.co2_model.predict(current_state_array.reshape(1, 1, -1))[0][0],
                'SO2': env.so2_model.predict(current_state_array.reshape(1, 1, -1))[0][0]
            }
            
            suggested_predictions = {
                'CB': env.cb_model.predict(suggested_state_array.reshape(1, 1, -1))[0][0],
                'CO2': env.co2_model.predict(suggested_state_array.reshape(1, 1, -1))[0][0],
                'SO2': env.so2_model.predict(suggested_state_array.reshape(1, 1, -1))[0][0]
            }
            
            st.subheader("Expected Improvements")
            cols = st.columns(3)
            
            # CB Production (higher is better)
            cols[0].metric(
                "CB Production",
                f"{suggested_predictions['CB']:.2f}",
                f"{suggested_predictions['CB'] - current_predictions['CB']:.2f}",
                help="Higher values are better"
            )
            
            # CO2 Emissions (lower is better)
            cols[1].metric(
                "CO2 Emissions",
                f"{suggested_predictions['CO2']:.2f}",
                f"{suggested_predictions['CO2'] - current_predictions['CO2']:.2f}",
                delta_color="inverse",
                help="Lower values are better"
            )
            
            # SO2 Emissions (lower is better)
            cols[2].metric(
                "SO2 Emissions",
                f"{suggested_predictions['SO2']:.2f}",
                f"{suggested_predictions['SO2'] - current_predictions['SO2']:.2f}",
                delta_color="inverse",
                help="Lower values are better"
            )
            
            # Add weighted improvement score
            total_improvement = (
                weights['CB'] * (suggested_predictions['CB'] - current_predictions['CB']) / current_predictions['CB'] -
                weights['CO2'] * (suggested_predictions['CO2'] - current_predictions['CO2']) / current_predictions['CO2'] -
                weights['SO2'] * (suggested_predictions['SO2'] - current_predictions['SO2']) / current_predictions['SO2']
            )
            
            st.metric(
                "Overall Weighted Improvement",
                f"{total_improvement * 100:.1f}%",
                help="Combines all objectives based on specified weights"
            )
            
            # Add button to apply suggestions
            if st.button("Apply Suggested Values"):
                for param, value in suggestions.items():
                    new_values[param] = value
                st.success("Applied suggested values! You can view them in the Parameter Control tab.")

    # Training section in sidebar
    with st.sidebar.expander("Agent Training", expanded=True):
        st.subheader("Training Parameters")
        
        # Training parameters
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            value=1e-4,
            help="Smaller = more stable but slower learning",
            key="sidebar_learning_rate"
        )
        
        n_steps = st.select_slider(
            "Steps per Update",
            options=[1024, 2048, 4096, 8192],
            value=4096,
            help="Larger = more stable learning",
            key="sidebar_n_steps"
        )
        
        n_epochs = st.slider(
            "Number of Epochs",
            min_value=3,
            max_value=10,
            value=5,
            help="How many times to reuse each batch of data",
            key="sidebar_n_epochs"
        )
        
        clip_range = st.select_slider(
            "Clip Range",
            options=[0.1, 0.15, 0.2, 0.25, 0.3],
            value=0.15,
            help="Smaller = more stable updates",
            key="sidebar_clip_range"
        )
        
        timesteps = st.select_slider(
            "Training Timesteps",
            options=[50000, 100000, 200000, 500000],
            value=100000,
            help="Total number of training steps",
            key="sidebar_timesteps"
        )
        
        # Try to load pre-trained agent
        model_path = f"models/rl_agent_reactor_{reactor_number}_multi"
        try:
            agent.load(model_path)
            st.success("Loaded trained agent")
        except:
            st.warning("No trained agent found. Using untrained agent.")

        if st.button("Train Agent"):
            # Create a new tab for training
            training_tab = st.tabs(["Training Progress"])[0]
            with training_tab:
                dashboard = create_training_dashboard()
                try:
                    with st.spinner("Configuring agent..."):
                        # Validate parameters
                        lr = float(learning_rate)
                        steps = int(n_steps)
                        epochs = int(n_epochs)
                        clip = float(clip_range)
                        total_steps = int(timesteps)
                        
                        if lr <= 0:
                            raise ValueError("Learning rate must be positive")
                        if steps <= 0:
                            raise ValueError("Steps must be positive")
                        if epochs <= 0:
                            raise ValueError("Epochs must be positive")
                        if clip <= 0:
                            raise ValueError("Clip range must be positive")
                        
                        # Configure agent with validated parameters
                        agent.configure_training(
                            learning_rate=lr,
                            n_steps=steps,
                            n_epochs=epochs,
                            clip_range=clip
                        )
                    
                    # Create callback with dashboard
                    callback = TrainingCallback(dashboard)
                    
                    with st.spinner("Training agent..."):
                        # Train agent
                        agent.train(total_timesteps=total_steps, callback=callback)
                        agent.save(model_path)
                    
                    st.success("Training completed successfully!")
                    
                except ValueError as ve:
                    st.error(f"Invalid parameter: {str(ve)}")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    logging.exception("Training error details:")
                    
            recommendations = analyze_training_metrics(dashboard['metrics_history'])
            if recommendations:
                st.subheader("Training Recommendations")
                for rec in recommendations:
                    st.markdown(rec)


if __name__ == "__main__":
    main()
