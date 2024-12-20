import os
from src.data_processing import load_and_preprocess_data, process_data_for_training, add_time_features, add_rolling_features, add_lag_features
from src.model import train_final_lstm_model_relu
from src.evaluation import validation_metrics_plot, predict_and_evaluate_model
from keras.models import load_model
import pandas as pd


def save_test_predictions(y_test, predictions, reactor, target_variable):
    """Save test data predictions for visualization"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create DataFrame with actual and predicted values
    test_df = pd.DataFrame({
        'actual': y_test.flatten(),
        'predicted': predictions.flatten()
    })
    
    # Save to CSV
    output_path = f'models/test_data_reactor_{reactor}_{target_variable.replace("|", "_")}.csv'
    test_df.to_csv(output_path, index=False)
    print(f"Saved test predictions to {output_path}")

def main():

    # Load and process data
    initial_cleaned_data = load_and_preprocess_data("data/Daten_juna.csv")

    def load_train_predict_evaluate(data_aux,reactor, target_variable):

        cleaned_data = data_aux.copy()
        cleaned_data = add_time_features(cleaned_data)
        cleaned_data = add_rolling_features(cleaned_data, target_col=target_variable)
        cleaned_data = add_lag_features(cleaned_data, target_col=target_variable)

        #reactor = 2
        #target_variable = "2|CB"
        if target_variable.endswith("|CB"):
            remove_variables = ["R{} CO2".format(reactor),"R{} SO2".format(reactor)]
        elif target_variable.endswith("CO2"):
            remove_variables = ["{}|CB".format(reactor),"R{} SO2".format(reactor)]
        else:
            remove_variables = ["{}|CB".format(reactor),"R{} CO2".format(reactor)]

        X_train_full, y_train_full, X_train, y_train, X_val, y_val, X_test, y_test, scaler, features = process_data_for_training(
            cleaned_data, reactor=reactor, seq_length=5, target_variable=target_variable, remove_variables=remove_variables
        )

        # Check if a trained model exists
        model_path = 'models/lstm_model_reactor_{}_target_{}.keras'.format(reactor, target_variable)
        if os.path.exists(model_path):
            print ("Model exists, loading model.")

            model = load_model(model_path)
        else:
            print ("Training, evaluating and saving model.")
            # Build and train the model
            model, history = train_final_lstm_model_relu(
                X_train_full, y_train_full, 
                X_val=None, y_val=None,
                epochs=100, 
                batch_size=32, 
                learning_rate=0.001, 
                lstm_units=[16], 
                optimizer_type="adam")
            model.save(model_path)
            # if we want validation metrics, add to the training step.
            #validation_metrics_plot(history,reactor,target_variable)

        predictions,y_test_actual = predict_and_evaluate_model(model, X_test, y_test, target_variable, scaler, features, reactor)

        # Save test predictions for visualization
        target_name = target_variable.split('|')[-1] if '|' in target_variable else target_variable.split()[-1]
        save_test_predictions(y_test_actual, predictions, reactor, target_name)

    # model all reactors for production, CO2 and SO2
    for r in [2, 3, 4, 5, 6, 7]:
        for t in ["{}|CB".format(r),"R{} CO2".format(r),"R{} SO2".format(r)]:
            load_train_predict_evaluate(initial_cleaned_data, r,t)     

if __name__ == "__main__":
    main()