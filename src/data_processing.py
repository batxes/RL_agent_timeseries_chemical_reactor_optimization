import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



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
        print (series.name)
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
                    and not col.endswith("CCT") 
                    and not col.endswith("CTD") 
                    and not col.endswith("FCC") 
                    and not col.endswith("SCT") 
                    #and not col.endswith("CO2") 
                    #and not col.endswith("SO2") 
                    and not col.startswith("KD")
                    and not col.startswith("KE")
                    and not col.startswith("LT7")
                    #and not col.endswith("Dampfmenge") 
                    and not col.endswith("Sorte"))]
    if len(remove_variables) > 0:
        for col in remove_variables:
            features.remove(col)
    print (features)

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


