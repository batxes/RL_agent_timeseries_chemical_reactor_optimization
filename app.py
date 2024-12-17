import streamlit as st
import numpy as np
from src.rl_agent import ReactorOptimizationAgent
from src.reactor_env import ChemicalReactorEnv
import logging
import os
def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    st.title("Chemical Reactor Optimization")

      # Add warning about missing models
    if not os.path.exists('src/models'):
        st.warning("""
        No trained models found. The application will run in demonstration mode with simulated predictions.
        
        To use actual predictions:
        1. Create a 'models' directory
        2. Add your trained LSTM models with the naming convention:
           - src/models/lstm_model_reactor_X_target_X|CB.keras
           - src/models/lstm_model_reactor_X_target_RX CO2.keras
           - src/models/lstm_model_reactor_X_target_RX SO2.keras
           (where X is the reactor number)
        """)
    # Sidebar controls
    reactor_number = st.sidebar.selectbox(
        "Select Reactor",
        options=[2, 3, 4, 5, 6, 7]
    )
    
    optimization_target = st.sidebar.selectbox(
        "Optimization Target",
        options=["CB", "CO2", "SO2"],
        format_func=lambda x: {
            "CB": "Maximize Production",
            "CO2": "Reduce CO2 Emissions",
            "SO2": "Reduce SO2 Emissions"
        }[x]
    )
    
    # Initialize agent
    agent = ReactorOptimizationAgent(reactor_number, optimization_target)
    
    # Try to load pre-trained agent
    model_path = f"models/rl_agent_reactor_{reactor_number}_{optimization_target}"
    try:
        agent.load(model_path)
        st.sidebar.success("Loaded trained agent")
    except:
        st.sidebar.warning("No trained agent found. Using untrained agent.")
    
    # Training section
    with st.sidebar.expander("Training Options"):
        timesteps = st.number_input("Training Timesteps", 
                                  min_value=1000, 
                                  max_value=100000, 
                                  value=50000, 
                                  step=1000)
        if st.button("Train Agent"):
            with st.spinner("Training agent..."):
                agent.train(total_timesteps=timesteps)
                agent.save(model_path)
            st.success("Training completed!")
    
    # Main interface
    st.header(f"Reactor {reactor_number} Control Panel")
    
    # Get current state from environment
    env = agent.env.envs[0]
    current_state = dict(zip(env.parameter_names, env.state))
    
    # Parameter controls
    st.subheader("Current Parameters")
    
    # Create tabs for parameter categories
    tabs = st.tabs([
        "Main Controls",
        "Temperature Controls",
        "Chemical Parameters",
        "Additional Controls"
    ])
    
    new_values = current_state.copy()
    
    # Main Controls
    with tabs[0]:
        cols = st.columns(2)
        for i, param in enumerate(['Erdgas', 'Konst.Stufe', 'Perlwasser', 'Regelstufe']):
            full_param = f"{reactor_number}|{param}"
            if full_param in current_state:
                with cols[i % 2]:
                    new_values[full_param] = st.slider(
                        param,
                        min_value=float(env.action_space.low[list(current_state.keys()).index(full_param)]),
                        max_value=float(env.action_space.high[list(current_state.keys()).index(full_param)]),
                        value=float(current_state[full_param])
                    )
    
    # Similar sections for other tabs...
    
    # Optimization section
    st.header("Optimization Suggestions")
    if st.button("Get Optimization Suggestions"):
        suggestions = agent.suggest_parameters(new_values)
        
        # Display suggestions with expected impact
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Suggested Changes")
            for param, value in suggestions.items():
                if abs(value - new_values[param]) > 0.1:  # Only show significant changes
                    st.write(f"{param}: {new_values[param]:.1f} → {value:.1f}")
        
        with col2:
            current_output = env._predict_output(np.array(list(new_values.values())))
            suggested_output = env._predict_output(np.array(list(suggestions.values())))
            
            metric_label = {
                "CB": "Production",
                "CO2": "CO2 Emissions",
                "SO2": "SO2 Emissions"
            }[optimization_target]
            
            st.metric(
                metric_label,
                f"{suggested_output:.2f}",
                f"{suggested_output - current_output:.2f}",
                delta_color="normal" if optimization_target == "CB" else "inverse"
            )

if __name__ == "__main__":
    main()