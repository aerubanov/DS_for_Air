import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import gpflow


data_file = 'DATA/processed/dataset.csv'


def convert_time(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = (
            data['timestamp'] - pd.to_datetime('2021-01-01', utc=True)
            )/pd.Timedelta(hours=1)
    return data


def get_data(file_path: str):
    data = pd.read_csv(file_path)
    data.spat.set_y_col('P1')
    qt = QuantileTransformer(
            output_distribution='normal',
            random_state=42,
            n_quantiles=100,
            )
    qt.fit(data.spat.y)

    data = data.spat.tloc['2021-01-01':'2021-01-30']
    data = data.dropna()
    data = data[['timestamp', 'lat', 'lon', 'P1', 'sds_sensor']]
    train_data = data.spat.tloc['2021-01-01':'2021-01-28']
    test_data = data.spat.tloc['2021-01-28':]

    # TODO replace by data.spat.y after solving issue 120
    train_data['P1'] = qt.transform(train_data.spat.y.values).flatten()
    test_data['P1'] = qt.transform(test_data.spat.y.values).flatten()

    train_data = convert_time(train_data)
    test_data = convert_time(test_data)

    return 
