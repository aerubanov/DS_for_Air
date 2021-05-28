import pandas as pd
from sklearn.preprocessing import QuantileTransformer


data_file = 'DATA/processed/dataset.csv'


def convert_x(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data[['timestamp', 'lon', 'lat']]
    data['timestamp'] = (
            data['timestamp'] - pd.to_datetime('2021-01-01', utc=True)
            )/pd.Timedelta(hours=1)
    return data.values


def get_data(file_path: str):
    data = pd.read_csv(file_path)
    data.spat.set_y_col('P1')
    qt = QuantileTransformer(
            output_distribution='normal',
            random_state=42,
            n_quantiles=100,
            )
    qt.fit(data.spat.y)

    data = data.spat.tloc['2021-01-01':'2021-02-01']
    data = data.dropna()
    train_data = data.spat.tloc['2021-01-01':'2021-01-29']
    test_data = data.spat.tloc['2021-01-29':]
    y = train_data.spat.y.values
    y = qt.transform(y).flatten()
    y_test = test_data.spat.y.values
    y_test = qt.transform(y_test).flatten()
    x = convert_x(train_data)
    x_test = convert_x(test_data)
    return x, y, x_test, y_test


if __name__ == '__main__':
    x, y, x_test, y_test = get_data(data_file)
