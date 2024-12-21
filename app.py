import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.rl_agent import ReactorOptimizationAgent, MultiObjectiveReactorEnv
from src.reactor_env import ChemicalReactorEnv
import logging
from stable_baselines3.common.callbacks import BaseCallback
import os
from matplotlib.gridspec import GridSpec

class TrainingCallback(BaseCallback):
    """Custom callback for displaying training progress"""
    
    def __init__(self, total_steps, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.progress_bar = None
    
    def _on_step(self) -> bool:
        """Update progress bar"""
        # Calculate progress
        progress = self.num_timesteps / self.total_steps
        
        # Update progress bar every 1000 steps
        if self.num_timesteps % 1000 == 0:
            if self.progress_bar is None:
                self.progress_bar = st.progress(0.0)
            self.progress_bar.progress(progress)
        
        return True

    def on_training_end(self) -> None:
        """Clean up progress bar"""
        if self.progress_bar is not None:
            self.progress_bar.progress(1.0)

def plot_training_complexity(agent):
    """Plot training complexity metrics using Matplotlib"""
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Training speed
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(agent.training_metrics['timesteps'], 
             agent.training_metrics['steps_per_second'])
    ax1.set_title('Training Speed')
    ax1.set_ylabel('Steps/Second')
    ax1.set_xlabel('Timesteps')
    
    # Parameter exploration
    ax2 = fig.add_subplot(gs[0, 1])
    for param, values in agent.training_metrics['parameter_updates'].items():
        if values:  # Only plot if we have data
            ax2.plot(agent.training_metrics['timesteps'][:len(values)], 
                    values, label=param.split('|')[-1])
    ax2.set_title('Parameter Exploration')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_xlabel('Timesteps')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Mean reward
    ax3 = fig.add_subplot(gs[1, 0])
    if agent.training_metrics['mean_reward']:  # Only plot if we have data
        ax3.plot(agent.training_metrics['timesteps'][:len(agent.training_metrics['mean_reward'])], 
                 agent.training_metrics['mean_reward'])
    ax3.set_title('Mean Reward')
    ax3.set_ylabel('Reward')
    ax3.set_xlabel('Timesteps')
    
    # Parameter activity heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    param_data = []
    for param in agent.raw_env.parameter_names:
        if param in agent.training_metrics['parameter_updates']:
            param_data.append(agent.training_metrics['parameter_updates'][param])
    if param_data:  # Only plot if we have data
        param_activity = np.array(param_data)
        im = ax4.imshow(param_activity, aspect='auto', cmap='viridis')
        ax4.set_title('Parameter Activity')
        ax4.set_ylabel('Parameters')
        ax4.set_yticks(range(len(agent.raw_env.parameter_names)))
        ax4.set_yticklabels([p.split('|')[-1] for p in agent.raw_env.parameter_names])
        plt.colorbar(im, ax=ax4)
    
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
    st.subheader("Training Plots")
    plots_container = st.container()  # This will hold our plot
    
    return {
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

    # Sidebar controls
    reactor_number = st.sidebar.selectbox(
        "Select Reactor",
        options=[2, 3, 4, 5, 6, 7]
    )
    
    # Multi-objective weights
    st.sidebar.subheader("Optimization Weights")
    weights = {
        'CB': st.sidebar.slider("Production (CB) Weight", 0.0, 1.0, 0.5, 0.1,
                               help="Higher value prioritizes production"),
        'CO2': st.sidebar.slider("CO2 Reduction Weight", 0.0, 1.0, 0.25, 0.1,
                                help="Higher value prioritizes CO2 reduction"),
        'SO2': st.sidebar.slider("SO2 Reduction Weight", 0.0, 1.0, 0.25, 0.1,
                                help="Higher value prioritizes SO2 reduction")
    }

    # Add variable selection in sidebar
    st.sidebar.subheader("Parameter Selection")
    
    # Group parameters for better organization
    parameter_groups = {
        "Main Controls": ['Erdgas', 'Konst.Stufe', 'Perlwasser', 'Regelstufe'],
        "Temperature Controls": ['VL Temp', 'Makeöl|Temperatur'],
        "Flow Controls": ['V-Luft', 'Fuelöl', 'Makeöl', 'Makeöl|Ventil'],
        "Process Parameters": ['CCT', 'CTD', 'FCC', 'SCT'],
        "Chemical Composition": ['C', 'H', 'N', 'O', 'S']
    }

    selected_parameters = {}
    with st.sidebar.expander("Select Parameters to Optimize", expanded=True):
        for group_name, params in parameter_groups.items():
            st.write(f"**{group_name}**")
            for param in params:
                full_param = f"{reactor_number}|{param}"
                selected_parameters[full_param] = st.checkbox(
                    param, 
                    value=True if param in ['Erdgas', 'V-Luft', 'Makeöl'] else False,  # Default selection
                    key=f"param_select_{full_param}"
                )
    
    # Get list of selected parameters
    active_parameters = [param for param, selected in selected_parameters.items() if selected]
    
    if len(active_parameters) < 1:
        st.warning("Please select at least one parameter to optimize")
        return
    elif len(active_parameters) > 10:
        st.warning("Consider selecting fewer parameters for more efficient optimization")
    
    st.sidebar.info(f"Optimizing {len(active_parameters)} parameters")
    
    # Show selected parameters
    with st.sidebar.expander("Selected Parameters", expanded=False):
        for param in active_parameters:
            st.write(f"- {param.split('|')[-1]}")
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
        # Initialize environment and agent with proper error handling
    try:
        # First try to load the models to verify they exist
        cb_model_path = f'models/lstm_model_reactor_{reactor_number}_target_{reactor_number}|CB.keras'
        co2_model_path = f'models/lstm_model_reactor_{reactor_number}_target_R{reactor_number} CO2.keras'
        so2_model_path = f'models/lstm_model_reactor_{reactor_number}_target_R{reactor_number} SO2.keras'
        
        if not all(os.path.exists(path) for path in [cb_model_path, co2_model_path, so2_model_path]):
            st.error("One or more required LSTM models not found. Please check model paths.")
            return
        
        # Initialize environment with existing models
        env = ChemicalReactorEnv(reactor_number, weights=weights, 
                                active_parameters=active_parameters)
        agent = ReactorOptimizationAgent(reactor_number, weights=weights, 
                                    active_parameters=active_parameters)
        
    except Exception as e:
        st.error(f"Error initializing environment and agent: {str(e)}")
        logging.exception("Initialization error details:")
        return
    


    # Debug info
    if st.checkbox("Show Debug Info"):
        st.write("Model Configuration:")
        st.write(f"- Expected timesteps: {env.expected_timesteps}")
        st.write(f"- Expected features: {env.expected_features}")
        st.write(f"- Active parameters: {len(active_parameters)}")
        
        # Show parameter mapping
        st.write("\nParameter Mapping:")
        for i, param in enumerate(env.parameter_names):
            st.write(f"{i}: {param}")

        st.write("Active Parameters:", active_parameters)
        st.write("State Shape:", env.observation_space.shape)
        st.write("Action Shape:", env.action_space.shape)
        if hasattr(env, 'state'):
            st.write("Current State (Active Parameters):", 
                    {param: val for param, val in zip(env.parameter_names, env.state)})
        st.write("Parameter Names:", env.parameter_names)
        st.write("State Shape:", env.state.shape)
        st.write("Parameter Config:", env.parameter_config)
        st.write("Optimization Weights:", weights)
        
        if not env.using_dummy_model:
            st.write("CB Model Shape:", env.cb_model.input_shape)
            st.write("CO2 Model Shape:", env.co2_model.input_shape)
            st.write("SO2 Model Shape:", env.so2_model.input_shape)

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

        # Training section
        st.subheader("Training")
        total_steps = st.number_input("Total timesteps", min_value=1000, value=10000, step=1000)
            
        if st.button("Train Agent"):
            with st.spinner("Training in progress..."):
                try:
                    # Create progress callback
                    progress_callback = TrainingCallback(total_steps=total_steps)
                    
                    # Train the agent
                    agent.train(total_timesteps=total_steps, callback=progress_callback)
                    
                    # Show completion message
                    st.success(f"Training completed in {agent.training_time:.1f} seconds!")
                    
                    # Show training metrics if available
                    if hasattr(agent, 'training_metrics'):
                        st.write("Training Metrics:")
                        st.write(f"- Average steps/second: {np.mean(agent.training_metrics['steps_per_second']):.1f}")
                        st.write(f"- Total steps: {agent.training_metrics['timesteps'][-1]}")
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    logging.exception("Training error details:")

    if st.button("Show Training Complexity Analysis"):
        if hasattr(agent, 'training_metrics') and agent.training_metrics['steps_per_second']:
            st.write("**Complexity Metrics:**")
            st.write(f"- Number of Active Parameters: {agent.complexity_metrics['action_space_size']}")
            st.write(f"- Total Possible Combinations: {agent.complexity_metrics['total_combinations']:.2e}")
            st.write(f"- Training Time: {agent.training_time:.2f} seconds")
            
            # Calculate efficiency metrics
            steps_per_param = total_steps / len(active_parameters)
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
        if not env.using_dummy_model:
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

            st.header("RL Agent Performance")
            if st.button("Evaluate Agent Performance"):
                with st.spinner("Evaluating agent..."):
                    fig, stats = plot_agent_performance(env, agent)
                    st.pyplot(fig)
                    
                    # Show performance metrics for all objectives
                    col1, col2, col3 = st.columns(3)
                    col1.metric("CB Improvement", f"{stats['mean_improvements']['CB']:.1f}%")
                    col2.metric("CO2 Reduction", f"{stats['mean_improvements']['CO2']:.1f}%",
                              delta_color="inverse")
                    col3.metric("SO2 Reduction", f"{stats['mean_improvements']['SO2']:.1f}%",
                              delta_color="inverse")
        else:
            st.warning("No trained models available. Using simulation mode.")
    
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
                                value=float(env.state[list(env.parameter_config.keys()).index(full_param)])
                            )
            
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


#def main():
#    logging.basicConfig(level=logging.INFO)
#    st.title("Chemical Reactor Optimization")
#
#    # Sidebar controls
#    reactor_number = st.sidebar.selectbox(
#        "Select Reactor",
#        options=[2, 3, 4, 5, 6, 7]
#    )
#    
#        # Replace optimization_target with weights
#    st.sidebar.subheader("Optimization Weights")
#    weights = {
#        'CB': st.sidebar.slider("Production (CB) Weight", 0.0, 1.0, 0.5, 0.1,
#                               help="Higher value prioritizes production"),
#        'CO2': st.sidebar.slider("CO2 Reduction Weight", 0.0, 1.0, 0.25, 0.1,
#                                help="Higher value prioritizes CO2 reduction"),
#        'SO2': st.sidebar.slider("SO2 Reduction Weight", 0.0, 1.0, 0.25, 0.1,
#                                help="Higher value prioritizes SO2 reduction")
#    }
#    
#    # Normalize weights to sum to 1
#    total_weight = sum(weights.values())
#    if total_weight > 0:
#        weights = {k: v/total_weight for k, v in weights.items()}
#    
#    # Initialize environment and agent with weights
#    env = MultiObjectiveReactorEnv(reactor_number, weights)
#    agent = ReactorOptimizationAgent(reactor_number, weights=weights)
#
#    # Debug info
#    if st.checkbox("Show Debug Info"):
#        st.write("Parameter Names:", env.parameter_names)
#        st.write("State Shape:", env.state.shape)
#        st.write("Parameter Config:", env.parameter_config)
#        
#        if not env.using_dummy_model:
#            st.write("Model Input Shape:", env.prediction_model.input_shape)
#            st.write("Model Output Shape:", env.prediction_model.output_shape)
#
#    # Training section in sidebar
#    with st.sidebar.expander("Agent Training", expanded=True):
#        st.subheader("Training Parameters")
#        
#        # Smaller learning rate for stability
#        learning_rate = st.select_slider(
#            "Learning Rate",
#            options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
#            value=1e-4,
#            help="Smaller = more stable but slower learning",
#            key="sidebar_learning_rate"
#        )
#        
#        # Larger batch size for better stability
#        n_steps = st.select_slider(
#            "Steps per Update",
#            options=[1024, 2048, 4096, 8192],
#            value=4096,
#            help="Larger = more stable learning",
#            key="sidebar_n_steps"
#        )
#        
#        # Fewer epochs to prevent overfitting
#        n_epochs = st.slider(
#            "Number of Epochs",
#            min_value=3,
#            max_value=10,
#            value=5,
#            help="How many times to reuse each batch of data",
#            key="sidebar_n_epochs"
#        )
#        
#        # Smaller clip range for stability
#        clip_range = st.select_slider(
#            "Clip Range",
#            options=[0.1, 0.15, 0.2, 0.25, 0.3],
#            value=0.15,
#            help="Smaller = more stable updates",
#            key="sidebar_clip_range"
#        )
#        
#        # More timesteps for better learning
#        timesteps = st.select_slider(
#            "Training Timesteps",
#            options=[50000, 100000, 200000, 500000],
#            value=100000,
#            help="Total number of training steps",
#            key="sidebar_timesteps"
#        )
#        
#        # Try to load pre-trained agent
#        model_path = f"models/rl_agent_reactor_{reactor_number}_{optimization_target}"
#        try:
#            agent.load(model_path)
#            st.success("Loaded trained agent")
#        except:
#            st.warning("No trained agent found. Using untrained agent.")
#
#        if st.button("Train Agent"):
#            # Create a new tab for training
#            training_tab = st.tabs(["Training Progress"])[0]
#            with training_tab:
#                dashboard = create_training_dashboard()
#                try:
#                    with st.spinner("Configuring agent..."):
#                        # Validate parameters
#                        lr = float(learning_rate)
#                        steps = int(n_steps)
#                        epochs = int(n_epochs)
#                        clip = float(clip_range)
#                        total_steps = int(timesteps)
#                        
#                        if lr <= 0:
#                            raise ValueError("Learning rate must be positive")
#                        if steps <= 0:
#                            raise ValueError("Steps must be positive")
#                        if epochs <= 0:
#                            raise ValueError("Epochs must be positive")
#                        if clip <= 0:
#                            raise ValueError("Clip range must be positive")
#                        
#                        # Configure agent with validated parameters
#                        agent.configure_training(
#                            learning_rate=lr,
#                            n_steps=steps,
#                            n_epochs=epochs,
#                            clip_range=clip
#                        )
#                    
#                    # Create callback with dashboard
#                    callback = TrainingCallback(dashboard)
#                    
#                    with st.spinner("Training agent..."):
#                        # Train agent
#                        agent.train(total_timesteps=total_steps, callback=callback)
#                        agent.save(model_path)
#                    
#                    st.success("Training completed successfully!")
#                    
#                except ValueError as ve:
#                    st.error(f"Invalid parameter: {str(ve)}")
#                except Exception as e:
#                    st.error(f"Error during training: {str(e)}")
#                    logging.exception("Training error details:")
#            recommendations = analyze_training_metrics(dashboard['metrics_history'])
#            if recommendations:
#                st.subheader("Training Recommendations")
#                for rec in recommendations:
#                    st.markdown(rec)
#    
#    # Create tabs
#    tabs = st.tabs(["Model Performance", "Parameter Control", "Optimization"])
#    
#    # Model Performance Tab
#    with tabs[0]:
#        st.header("LSTM Model Performance")
#        if not env.using_dummy_model:
#            # Show performance for all three objectives
#            for objective in ['CB', 'CO2', 'SO2']:
#                st.subheader(f"{objective} Model Performance")
#                fig = plot_model_performance(reactor_number, objective)
#                if fig:
#                    st.pyplot(fig)
#                    
#                    # Add performance metrics
#                    test_data = pd.read_csv(f'models/test_data_reactor_{reactor_number}_{optimization_target}.csv')
#                    mse = np.mean((test_data['actual'] - test_data['predicted'])**2)
#                    r2 = np.corrcoef(test_data['actual'], test_data['predicted'])[0,1]**2
#                    
#                    col1, col2 = st.columns(2)
#                    col1.metric("Mean Squared Error", f"{mse:.4f}")
#                    col2.metric("R² Score", f"{r2:.4f}")
#
#            st.header("RL Agent Performance")
#            if st.button("Evaluate Agent Performance"):
#                with st.spinner("Evaluating agent..."):
#                    fig, mean_reward, mean_improvement = plot_agent_performance(env, agent)
#                    st.pyplot(fig)
#                    
#                    # Show performance metrics
#                    col1, col2 = st.columns(2)
#                    col1.metric("Mean Episode Reward", f"{mean_reward:.2f}")
#                    col2.metric(f"Mean {optimization_target} Improvement", 
#                              f"{mean_improvement:.1f}%",
#                              delta_color="normal" if optimization_target == "CB" else "inverse")
#        else:
#            st.warning("No trained model available. Using simulation mode.")
#    
#    # Parameter Control Tab
#    with tabs[1]:
#        st.header("Parameter Control")
#        
#        # Group parameters by category
#        parameter_groups = {
#            "Main Controls": ['Erdgas', 'Konst.Stufe', 'Perlwasser', 'Regelstufe'],
#            "Temperature Controls": ['VL Temp', 'Makeöl|Temperatur'],
#            "Flow Controls": ['V-Luft', 'Fuelöl', 'Makeöl', 'Makeöl|Ventil'],
#            "Process Parameters": ['CCT', 'CTD', 'FCC', 'SCT'],
#            "Chemical Composition": ['C', 'H', 'N', 'O', 'S']
#        }
#        
#        # Create form for parameters
#        with st.form("parameter_form"):
#            new_values = {}
#            
#            for group_name, params in parameter_groups.items():
#                st.subheader(group_name)
#                cols = st.columns(2)
#                for i, param in enumerate(params):
#                    full_param = f"{reactor_number}|{param}"
#                    if full_param in env.parameter_config:
#                        with cols[i % 2]:
#                            min_val, max_val = env.parameter_config[full_param]
#                            new_values[full_param] = st.slider(
#                                param,
#                                min_value=float(min_val),
#                                max_value=float(max_val),
#                                value=float(env.state[list(env.parameter_config.keys()).index(full_param)])
#                            )
#            
#            # Predict button
#            if st.form_submit_button("Predict Output"):
#                state_array = np.array([new_values[param] for param in env.parameter_names])
#                predictions = {
#                    'CB': env.cb_model.predict(state_array.reshape(1, 1, -1))[0][0],
#                    'CO2': env.co2_model.predict(state_array.reshape(1, 1, -1))[0][0],
#                    'SO2': env.so2_model.predict(state_array.reshape(1, 1, -1))[0][0]
#                }
#                
#                cols = st.columns(3)
#                cols[0].metric("Predicted CB", f"{predictions['CB']:.2f}")
#                cols[1].metric("Predicted CO2", f"{predictions['CO2']:.2f}")
#                cols[2].metric("Predicted SO2", f"{predictions['SO2']:.2f}")
#    
#    # Optimization Tab
#    with tabs[2]:
#        st.header("Parameter Optimization")
#        
#        if st.button("Get Optimization Suggestions"):
#            # Get current state and suggestions
#            current_state = {param: new_values.get(param, env.state[i]) 
#                           for i, param in enumerate(env.parameter_names)}
#            suggestions = agent.suggest_parameters(current_state)
#            
#            # Display current vs suggested values
#            st.subheader("Parameter Suggestions")
#            
#            for group_name, params in parameter_groups.items():
#                st.write(f"\n**{group_name}**")
#                cols = st.columns(3)
#                cols[0].write("Parameter")
#                cols[1].write("Current")
#                cols[2].write("Suggested")
#                
#                for param in params:
#                    full_param = f"{reactor_number}|{param}"
#                    if full_param in suggestions:
#                        cols = st.columns(3)
#                        cols[0].write(param)
#                        cols[1].write(f"{current_state[full_param]:.2f}")
#                        cols[2].write(f"{suggestions[full_param]:.2f}")
#            
#            # Show expected improvement
#            current_output = env._predict_output(np.array(list(current_state.values())))
#            suggested_output = env._predict_output(np.array(list(suggestions.values())))
#            
#            st.metric(
#                f"Expected {optimization_target}",
#                f"{suggested_output:.2f}",
#                f"{suggested_output - current_output:.2f}",
#                delta_color="normal" if optimization_target == "CB" else "inverse"
#            )
#            
#            # Add button to apply suggestions
#            if st.button("Apply Suggested Values"):
#                for param, value in suggestions.items():
#                    new_values[param] = value
#                st.success("Applied suggested values! You can view them in the Parameter Control tab.")

if __name__ == "__main__":
    main()
