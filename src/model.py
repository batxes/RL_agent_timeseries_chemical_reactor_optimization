from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def train_final_lstm_model_relu(X_train, y_train, X_val=None, y_val=None,
                           lstm_units=[32], 
                           recurrent_dropout=0.25, 
                           dropout=0.5, 
                           learning_rate=0.001, 
                           batch_size=32, 
                           epochs=100, 
                           optimizer_type='adam',
                           ):
    
    # Model architecture
    inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = inputs

    # Add LSTM and Dropout layers for each value in lstm_units
    if len(lstm_units) > 1:
        for units in lstm_units[:-1]:
            x = layers.LSTM(units, recurrent_dropout=recurrent_dropout, return_sequences=True, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
    
    # Remove return_sequences for the last LSTM layer
    x = layers.LSTM(lstm_units[-1], recurrent_dropout=recurrent_dropout)(x)
    x = layers.Dropout(dropout)(x)
    
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    if optimizer_type.lower() == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Optimizer must be either 'adam' or 'rmsprop'")
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    print(model.summary())
    
    if X_val is not None:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
        )
    else:
    # Train the model without validation data
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs
        )
    
    return model, history

