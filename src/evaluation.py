import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

def validation_metrics_plot(history,reactor, target_variable):

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.legend()
    ax2.set_title('Training and Validation MAE')

    #plt.show()
    plt.savefig("figures/validation_model_reactor_{}_target_{}".format(reactor, target_variable))

    print ("Best val loss: {}".format(min(history.history["val_loss"])))

def predict_and_evaluate_model(model, X_test, y_test, target_variable, scaler, features, reactor):

    logging.basicConfig(
    filename='logs/Evaluation.log',  # Log to a file named 'app.log'
    level=logging.INFO,  # Set the log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

    predictions = model.predict(X_test)
    predictions_normalized = predictions

    cb_index = features.index(target_variable)
    
    # Inverse transform the predictions and actual values
    predictions = scaler.inverse_transform(np.column_stack((np.zeros((len(predictions), cb_index)), 
                                                            predictions, 
                                                            np.zeros((len(predictions), len(features)-cb_index-1)))))[:, cb_index]

    y_test_actual = scaler.inverse_transform(np.column_stack((np.zeros((len(y_test), cb_index)), 
                                                                y_test.reshape(-1, 1), 
                                                                np.zeros((len(y_test), len(features)-cb_index-1)))))[:, cb_index]

    # Calculate metrics
    mse = mean_squared_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)
    mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R2 Score: {r2}')
    print(f'MAPE: {mape}%')

    logging.info(f'-------- Reactor: {reactor} Target: {target_variable}')
    logging.info(f'MSE: {mse}')
    logging.info(f'RMSE: {rmse}')
    logging.info(f'MAE: {mae}')
    logging.info(f'R2 Score: {r2}')
    logging.info(f'MAPE: {mape}%')
    logging.info('\n')


    # Sort the actual test data and predictions
    sorted_indices = np.argsort(y_test_actual)
    y_test_sorted = y_test_actual[sorted_indices]
    predictions_sorted = predictions[sorted_indices]

    # Create a figure with 5 subplots
    fig = plt.figure(figsize=(12, 10))

    ax2 = fig.add_subplot(2, 1, 1)
    ax2.plot(y_test_actual, label='Actual', color='blue', alpha=0.7)
    ax2.plot(predictions, label='Predicted', color='red', alpha=0.7)
    ax2.set_title('Actual vs Predicted Values')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.legend()

    ax3 = fig.add_subplot(2, 1, 2)
    residuals = y_test_actual - predictions
    ax3.scatter(predictions, residuals)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Plot')

    plt.tight_layout()
    plt.savefig("figures/evaluation_model_reactor_{}_target_{}.pdf".format(reactor, target_variable), dpi=300)

    #plt.show()

    return predictions, y_test_actual