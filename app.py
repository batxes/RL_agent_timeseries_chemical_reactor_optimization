import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.rl_agent import ReactorOptimizationAgent
from src.reactor_env import ChemicalReactorEnv
import logging
import os

def plot_agent_training_history(history):
    """Plot agent training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    ax1.plot(history['rollout/ep_rew_mean'], label='Mean Episode Reward')
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Reward')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history['train/policy_loss'], label='Policy Loss', alpha=0.7)
    ax2.plot(history['train/value_loss'], label='Value Loss', alpha=0.7)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Updates')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    return fig

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

def main():

    logging.basicConfig(level=logging.INFO)

    # In the Parameter Control Tab, before prediction
    if st.checkbox("Show Debug Info"):
        st.write("Parameter Names:", env.parameter_names)
        st.write("State Shape:", env.state.shape)
        st.write("Parameter Config:", env.parameter_config)
        
        if not env.using_dummy_model:
            st.write("Model Input Shape:", env.prediction_model.input_shape)
            st.write("Model Output Shape:", env.prediction_model.output_shape)
    st.title("Chemical Reactor Optimization")
    
    # Sidebar controls
    reactor_number = st.sidebar.selectbox(
        "Select Reactor",
        options=[2, 3, 4, 5, 6, 7]
    )
    
    optimization_target = st.sidebar.selectbox(
        "Optimization Target",
        options=["CB", "CO2", "SO2"],
        format_func=lambda x: {
            "CB": "Production (CB)",
            "CO2": "CO2 Emissions",
            "SO2": "SO2 Emissions"
        }[x]
    )
    
    # Initialize environment and agent
    env = ChemicalReactorEnv(reactor_number, optimization_target)
    agent = ReactorOptimizationAgent(reactor_number, optimization_target)
    
    # Training section in sidebar
    with st.sidebar.expander("Agent Training"):
        timesteps = st.number_input("Training Timesteps", 
                                  min_value=1000, 
                                  max_value=100000, 
                                  value=50000, 
                                  step=1000)
        
        # Try to load pre-trained agent
        model_path = f"models/rl_agent_reactor_{reactor_number}_{optimization_target}"
        try:
            agent.load(model_path)
            st.success("Loaded trained agent")
        except:
            st.warning("No trained agent found. Using untrained agent.")
        
        if st.button("Train Agent"):
            with st.spinner("Training agent..."):
                history = agent.train(total_timesteps=timesteps)
                agent.save(model_path)
            st.success("Training completed!")
            
            # Show training metrics
            st.subheader("Training Metrics")
            fig = plot_agent_training_history(history)
            st.pyplot(fig)
            
            # Show final metrics
            col1, col2 = st.columns(2)
            final_reward = history['rollout/ep_rew_mean'][-1]
            success_rate = history.get('rollout/success_rate', [0])[-1]
            col1.metric("Final Mean Reward", f"{final_reward:.2f}")
            col2.metric("Success Rate", f"{success_rate:.1%}")
    
    # Create tabs
    tabs = st.tabs(["Model Performance", "Parameter Control", "Optimization"])
    
    # Model Performance Tab
    with tabs[0]:
        st.header("LSTM Model Performance")
        if not env.using_dummy_model:
            fig = plot_model_performance(reactor_number, optimization_target)
            if fig:
                st.pyplot(fig)
                
                # Add performance metrics
                test_data = pd.read_csv(f'models/test_data_reactor_{reactor_number}_{optimization_target}.csv')
                mse = np.mean((test_data['actual'] - test_data['predicted'])**2)
                r2 = np.corrcoef(test_data['actual'], test_data['predicted'])[0,1]**2
                
                col1, col2 = st.columns(2)
                col1.metric("Mean Squared Error", f"{mse:.4f}")
                col2.metric("R² Score", f"{r2:.4f}")

            st.header("RL Agent Performance")
            if st.button("Evaluate Agent Performance"):
                with st.spinner("Evaluating agent..."):
                    fig, mean_reward, mean_improvement = plot_agent_performance(env, agent)
                    st.pyplot(fig)
                    
                    # Show performance metrics
                    col1, col2 = st.columns(2)
                    col1.metric("Mean Episode Reward", f"{mean_reward:.2f}")
                    col2.metric(f"Mean {optimization_target} Improvement", 
                              f"{mean_improvement:.1f}%",
                              delta_color="normal" if optimization_target == "CB" else "inverse")
        else:
            st.warning("No trained model available. Using simulation mode.")
    
    # Parameter Control Tab
    with tabs[1]:
        st.header("Parameter Control")
        
        # Group parameters by category
        parameter_groups = {
            "Main Controls": ['Erdgas', 'Konst.Stufe', 'Perlwasser', 'Regelstufe'],
            "Temperature Controls": ['VL Temp', 'Makeöl|Temperatur'],
            "Chemical Parameters": ['C', 'H', 'N', 'O', 'S'],
            "Additional Controls": ['V-Luft', 'Fuelöl', 'Makeöl', 'Makeöl|Ventil', 
                                  'CCT', 'CTD', 'FCC', 'SCT']
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
                predicted_output = env._predict_output(state_array)
                
                st.metric(
                    f"Predicted {optimization_target}",
                    f"{predicted_output:.2f}"
                )
    
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
            
            # Show expected improvement
            current_output = env._predict_output(np.array(list(current_state.values())))
            suggested_output = env._predict_output(np.array(list(suggestions.values())))
            
            st.metric(
                f"Expected {optimization_target}",
                f"{suggested_output:.2f}",
                f"{suggested_output - current_output:.2f}",
                delta_color="normal" if optimization_target == "CB" else "inverse"
            )
            
            # Add button to apply suggestions
            if st.button("Apply Suggested Values"):
                for param, value in suggestions.items():
                    new_values[param] = value
                st.success("Applied suggested values! You can view them in the Parameter Control tab.")

if __name__ == "__main__":
    main()

#
#import streamlit as st
#import numpy as np
#from src.rl_agent import ReactorOptimizationAgent
#from src.reactor_env import ChemicalReactorEnv
#import logging
#import os
#def main():
#    # Configure logging
#    logging.basicConfig(level=logging.INFO)
#    st.title("Chemical Reactor Optimization")
#
#      # Add warning about missing models
#    if not os.path.exists('src/models'):
#        st.warning("""
#        No trained models found. The application will run in demonstration mode with simulated predictions.
#        
#        To use actual predictions:
#        1. Create a 'models' directory
#        2. Add your trained LSTM models with the naming convention:
#           - src/models/lstm_model_reactor_X_target_X|CB.keras
#           - src/models/lstm_model_reactor_X_target_RX CO2.keras
#           - src/models/lstm_model_reactor_X_target_RX SO2.keras
#           (where X is the reactor number)
#        """)
#    # Sidebar controls
#    reactor_number = st.sidebar.selectbox(
#        "Select Reactor",
#        options=[2, 3, 4, 5, 6, 7]
#    )
#    
#    optimization_target = st.sidebar.selectbox(
#        "Optimization Target",
#        options=["CB", "CO2", "SO2"],
#        format_func=lambda x: {
#            "CB": "Maximize Production",
#            "CO2": "Reduce CO2 Emissions",
#            "SO2": "Reduce SO2 Emissions"
#        }[x]
#    )
#    
#    # Initialize agent
#    agent = ReactorOptimizationAgent(reactor_number, optimization_target)
#    
#    # Try to load pre-trained agent
#    model_path = f"models/rl_agent_reactor_{reactor_number}_{optimization_target}"
#    try:
#        agent.load(model_path)
#        st.sidebar.success("Loaded trained agent")
#    except:
#        st.sidebar.warning("No trained agent found. Using untrained agent.")
#    
#    # Training section
#    with st.sidebar.expander("Training Options"):
#        timesteps = st.number_input("Training Timesteps", 
#                                  min_value=1000, 
#                                  max_value=100000, 
#                                  value=50000, 
#                                  step=1000)
#        if st.button("Train Agent"):
#            with st.spinner("Training agent..."):
#                agent.train(total_timesteps=timesteps)
#                agent.save(model_path)
#            st.success("Training completed!")
#    
#    # Main interface
#    st.header(f"Reactor {reactor_number} Control Panel")
#    
#    # Get current state from environment
#    env = agent.env.envs[0]
#    current_state = dict(zip(env.parameter_names, env.state))
#    
#    # Parameter controls
#    st.subheader("Current Parameters")
#    
#    # Create tabs for parameter categories
#    tabs = st.tabs([
#        "Main Controls",
#        "Temperature Controls",
#        "Chemical Parameters",
#        "Additional Controls"
#    ])
#    
#    new_values = current_state.copy()
#    
#    # Main Controls
#    with tabs[0]:
#        cols = st.columns(2)
#        for i, param in enumerate(['Erdgas', 'Konst.Stufe', 'Perlwasser', 'Regelstufe']):
#            full_param = f"{reactor_number}|{param}"
#            if full_param in current_state:
#                with cols[i % 2]:
#                    new_values[full_param] = st.slider(
#                        param,
#                        min_value=float(env.action_space.low[list(current_state.keys()).index(full_param)]),
#                        max_value=float(env.action_space.high[list(current_state.keys()).index(full_param)]),
#                        value=float(current_state[full_param])
#                    )
#    
#    # Similar sections for other tabs...
#    
#    # Optimization section
#    st.header("Optimization Suggestions")
#    if st.button("Get Optimization Suggestions"):
#        suggestions = agent.suggest_parameters(new_values)
#        
#        # Display suggestions with expected impact
#        col1, col2 = st.columns(2)
#        with col1:
#            st.subheader("Suggested Changes")
#            for param, value in suggestions.items():
#                if abs(value - new_values[param]) > 0.1:  # Only show significant changes
#                    st.write(f"{param}: {new_values[param]:.1f} → {value:.1f}")
#        
#        with col2:
#            current_output = env._predict_output(np.array(list(new_values.values())))
#            suggested_output = env._predict_output(np.array(list(suggestions.values())))
#            
#            metric_label = {
#                "CB": "Production",
#                "CO2": "CO2 Emissions",
#                "SO2": "SO2 Emissions"
#            }[optimization_target]
#            
#            st.metric(
#                metric_label,
#                f"{suggested_output:.2f}",
#                f"{suggested_output - current_output:.2f}",
#                delta_color="normal" if optimization_target == "CB" else "inverse"
#            )
#
#if __name__ == "__main__":
#    main()