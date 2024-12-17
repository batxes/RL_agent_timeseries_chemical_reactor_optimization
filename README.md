# Chemical Reactor Optimization

A Streamlit application that uses reinforcement learning to optimize chemical reactor parameters. The system can optimize for either maximizing production (CB) or minimizing emissions (CO2, SO2) across multiple reactors.

## Explanation

A producer of chemicals. We have a timeseries for this year, resolution is 1h, the data is available for 6 reactors, so in total 37728 samples. Each sample has a number of values, e.g. for reactor 2:

2|CB 	2|Erdgas 	2|Konst.Stufe 	2|Perlwasser 	2|Regelstufe 	2|Sorte 	2|V-Luft 	2|VL Temp 	2|Fuelöl 	2|Makeöl 	2|Makeöl|Temperatur 	2|Makeöl|Ventil 	2|CCT 	2|CTD 	2|FCC 	2|SCT 	2|C 	2|H 	2|N 	2|O 	2|S
R2 CO2
R2 SO2
and there are some values that are common for all 6 reactors:
KD|Dampfmenge 	KD|Restgasmenge 	KD|NOx 	KD|Rauchgasmenge 	KD|SO2 	KE|Dampfmenge 	KE|Restgasmenge 	KE|NOx 	KE|Rauchgasmenge 	KE|SO2

most interesting would be to try to predict output values from the input values. Important outputs are:
- CB (the product)
- SO2
- CO2
- Dampfmenge (steam)
- Rauchgasmenge (tail gas, all reactors combined)

input is basically everything else except the values for all reactors:
KD|Dampfmenge 	KD|Restgasmenge 	KD|NOx 	KD|Rauchgasmenge 	KD|SO2 	KE|Dampfmenge 	KE|Restgasmenge 	KE|NOx 	KE|Rauchgasmenge 	KE|SO2


## Features

- **Interactive Web Interface**: Built with Streamlit for easy parameter control and visualization
- **Multiple Optimization Targets**:
  - Production optimization (CB)
  - CO2 emissions reduction
  - SO2 emissions reduction
- **Support for Multiple Reactors**: Handles reactors 2-7
- **Real-time Parameter Adjustment**:
  - Erdgas (Natural gas)
  - Konst.Stufe (Constant stage)
  - Perlwasser (Pearl water)
  - Regelstufe (Control stage)
  - V-Luft (Ventilation air)
  - VL Temp (Temperature)
  - Fuelöl (Fuel oil)
  - Makeöl (Make oil)
  - And more...
- **Reinforcement Learning Agent**: Uses PPO (Proximal Policy Optimization) for parameter optimization
- **Safety Features**: Includes penalties for extreme values and rapid changes

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/chemical-reactor-optimization.git
cd chemical-reactor-optimization

2. Create and activate virtual environment:

Using venv
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
Or using conda
conda create -n chemical python=3.9
conda activate chemical

3. Install requirements:
pip install -r requirements.txt


## Usage

1. Run the Streamlit application:
streamlit run app.py
2. Open your browser and go to:
http://localhost:8501


3. In the application:
   - Select a reactor (2-7)
   - Choose optimization target (Production, CO2, or SO2)
   - Adjust parameters manually or use the optimization suggestions
   - Train the RL agent if needed

## Project Structure
chemical-optimization/
├── app.py # Streamlit application
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── src/ # Source code
├── init.py
├── reactor_env.py # Gym environment for reactor
├── rl_agent.py # RL agent implementation
└── models/ # Directory for trained models

## Model Requirements

The application can work in two modes:
1. **With trained models**: Place your trained LSTM models in `src/models/` with the naming convention:
   - `lstm_model_reactor_X_target_X|CB.keras`
   - `lstm_model_reactor_X_target_RX CO2.keras`
   - `lstm_model_reactor_X_target_RX SO2.keras`
   (where X is the reactor number)

2. **Without models**: The application will run in demonstration mode with simulated predictions

## Development

### Requirements
- Python 3.9+
- PyTorch
- Tensorflow
- Streamlit
- Gymnasium
- Stable-baselines3

### Environment Variables
None required for basic operation.

### Running Tests

TODO: Add testing instructions

## Docker Support

1. Build the Docker image:
docker build -t chemical-optimization .

2. Run the container:
docker run -p 8501:8501 chemical-optimization

## Roadmap

- [ ] Add support for multi-objective optimization
- [ ] Implement more sophisticated reward functions
- [ ] Add visualization of parameter relationships
- [ ] Improve safety constraints