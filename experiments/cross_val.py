from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import numpy as np

data_file = 'DATA/processed/dataset.csv'


def get_data(file_path: str):
    data = pd.read_csv(file_path)
    data = data.groupby('timestamp').median()
    data = data.reset_index()
    data = data.interpolate()
    data.spat.set_y_col('P1')
    qt = QuantileTransformer(
            output_distribution='normal',
            random_state=42,
            n_quantiles=100,
            )
    qt.fit(data.spat.y)

    data = data.spat.tloc['2021-01-01':'2021-03-1']

    x = np.arange(len(data))[:, None]
    y = data.spat.y.values
    y = qt.transform(y).flatten()
    return x, y


if __name__ == '__main__':
    x, y = get_data(data_file)
    n_splits = round((len(x)-20*24)/24)
    cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=20*24, test_size=24)
    splits = cv.split(x)
    train_index, test_index = next(splits)
    print(min(train_index), max(train_index), min(test_index), max(test_index))
    for train_index, test_index in splits:
        print(min(train_index), max(train_index), min(test_index), max(test_index))
