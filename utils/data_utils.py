import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(['120_종가','120_거래량'], axis=1, inplace=True)
    df['시간'] = pd.to_datetime(df['시간'], format='%m/%d,%H:%M')
    market_open_time = pd.to_datetime('09:00', format='%H:%M').time()
    break_start_time = pd.to_datetime('15:20', format='%H:%M').time()
    break_end_time = pd.to_datetime('15:30', format='%H:%M').time()

    elapsed_time = (df['시간'].dt.hour - market_open_time.hour) * 60 + (df['시간'].dt.minute - market_open_time.minute)
    break_mask = (df['시간'].dt.time >= break_start_time) & (df['시간'].dt.time <= break_end_time)
    df['시간'] = elapsed_time.mask(break_mask, -1)

    df = df.iloc[::-1].reset_index(drop=True)
    str_columns = df.select_dtypes(include=['object']).columns
    df[str_columns] = df[str_columns].applymap(lambda x: x.replace(',', '') if isinstance(x, str) else x)
    df[str_columns] = df[str_columns].apply(pd.to_numeric, errors='coerce')

    return df

def scale(value, scale_factor):
    return np.array(value, dtype=float) / scale_factor

def min_max_scale(data, min, max):
    return (data - min) / (max - min)

def z_score_normalize(data):
    return (data - np.mean(data)) / np.std(data)