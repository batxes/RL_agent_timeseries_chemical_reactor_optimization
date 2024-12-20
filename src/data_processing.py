import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Time-based features
def add_time_features(df):
    # Convert Bezeichnung to datetime if not already
    df['datetime'] = pd.to_datetime(df['Bezeichnung'], format='%d/%m/%Y %H:%M:%S')
    
    # Extract time components
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    return df

def add_rolling_features(df, target_col, windows=[3, 6, 12, 24]):
    for window in windows:
        # Calculate rolling features
        df[f'{target_col}_rolling_mean_{window}h'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}h'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}h'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}h'] = df[target_col].rolling(window=window).max()
        
        # Fill NaN values
        # For mean and std, use the actual value
        df[f'{target_col}_rolling_mean_{window}h'] = df[f'{target_col}_rolling_mean_{window}h'].fillna(df[target_col])
        df[f'{target_col}_rolling_std_{window}h'] = df[f'{target_col}_rolling_std_{window}h'].fillna(0)
        
        # For min/max, use the actual value
        df[f'{target_col}_rolling_min_{window}h'] = df[f'{target_col}_rolling_min_{window}h'].fillna(df[target_col])
        df[f'{target_col}_rolling_max_{window}h'] = df[f'{target_col}_rolling_max_{window}h'].fillna(df[target_col])
    
    return df

def add_lag_features(df, target_col, lags=[1, 2, 3, 6, 12, 24]):
    for lag in lags:
        # Create lag feature
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
        
        # Fill NaN values with the mean of the target column
        df[f'{target_col}_lag_{lag}h'] = df[f'{target_col}_lag_{lag}h'].fillna(df[target_col].mean())
        
        # Alternative: forward fill
        # df[f'{target_col}_lag_{lag}h'] = df[f'{target_col}_lag_{lag}h'].ffill()
    
    return df

def load_and_preprocess_data(data_path="data/Daten_juna.csv"):
    df = pd.read_csv(data_path,sep=",")
    df = df.iloc[1:,:] # remove first row, there are two headers
    def safe_numeric_convert(series):
        try:
            return pd.to_numeric(series)
        except ValueError:
            return series
    
    list_of_failed_measurements = []

    def check_non_numeric(series):
        non_numeric = series[pd.to_numeric(series, errors='coerce').isna() & series.notna()]
        return (len(non_numeric),set(non_numeric)),list(set(non_numeric))

    non_numeric,values = check_non_numeric(df["R2 CO2"])
    list_of_failed_measurements.extend(values)

    non_numeric,values = check_non_numeric(df["3|Fuelöl"])
    list_of_failed_measurements.extend(values)

    non_numeric,values = check_non_numeric(df["2|CCT"])
    list_of_failed_measurements.extend(values)

    non_numeric,values = check_non_numeric(df["LT7|S"])
    list_of_failed_measurements.extend(values)

    non_numeric,values = check_non_numeric(df["4|Perlwasser"])
    list_of_failed_measurements.extend(values)

    non_numeric,values = check_non_numeric(df["7|Konst.Stufe"])
    list_of_failed_measurements.extend(values)

    mask = df.isin(list_of_failed_measurements).any(axis=1)
    df_cleaned = df[~mask]

    for col in df.columns:
        df_cleaned[col] = safe_numeric_convert(df_cleaned[col])
        #print(f"{col:<20}{df_cleaned[col].dtype}")


    return df_cleaned


def process_data_for_training(df_cleaned, reactor=2, seq_length=5, target_variable="2|CB", remove_variables=["R2 CO2", "R2 SO2"]):
    
    features = [col for col in df_cleaned.columns 
                if ((col.startswith(str(reactor)) or col.startswith("R"+str(reactor))) 
                    #and not col.endswith("CCT") 
                    #and not col.endswith("CTD") 
                    #and not col.endswith("FCC") 
                    #and not col.endswith("SCT") 
                    #and not col.endswith("CO2") 
                    #and not col.endswith("SO2") 
                    and not col.startswith("KD")
                    and not col.startswith("KE")
                    and not col.startswith("LT7")
                    and not col.endswith("Dampfmenge") 
                    and not col.endswith("Sorte"))]
    if len(remove_variables) > 0:
        for col in remove_variables:
            features.remove(col)

    # Data preprocessing done. Now we will process for feeding the training.
    data = df_cleaned[features].values
    df_cleaned[features].to_csv("data/df_cleaned_for_reactor_{}_target_{}.tsv".format(reactor,target_variable), sep="\t")

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(data)

    cb_index = features.index(target_variable)

    # Create sequences to give temporal context to the model
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length), :])
            y.append(data[i + seq_length, cb_index]) 
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, seq_length)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, shuffle=False)

    return X_train_full, y_train_full, X_train, y_train, X_val, y_val, X_test, y_test, scaler, features


