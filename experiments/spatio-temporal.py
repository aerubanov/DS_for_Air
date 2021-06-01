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

    return train_data, test_data


def train_model(data: pd.DataFrame, M=200) -> gpflow.models.sgpr.SGPR:
    x = data[['timestamp', 'lon', 'lat']].values
    y = data['P1'].values[:, None]

    spat_cov = gpflow.kernels.Matern32(
            variance=1,
            lengthscales=0.1,
            active_dims=[1, 2],
            )

    mt = gpflow.kernels.Matern32(variance=1, lengthscales=24, active_dims=[0])
    pk = gpflow.kernels.Periodic(
            gpflow.kernels.SquaredExponential(
                lengthscales=24*1.5,
                active_dims=[0],
                ),
            period=24,
            )
    time_cov = mt * pk

    cov = spat_cov * time_cov

    Z = x[np.random.permutation(x.shape[0])[0:M], :]
    model = gpflow.models.sgpr.SGPR((x, y), cov, Z)

    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
            model.training_loss,
            model.trainable_variables,
            options={'iprint': 50, 'maxiter': 600},
            )
    return model


if __name__ == '__main__':
    df_train, df_test = get_data(data_file)
    print(df_train)
    print(df_test)
    print(df_train.timestamp.max(), df_test.timestamp.min())
    df_train.groupby('timestamp').mean().reindex().P1.plot()
    plt.show()
    df_test.groupby('timestamp').mean().reindex().P1.plot()
    plt.show()
    model = train_model(df_train, M=400)
    gpflow.utilities.print_summary(model)

    x_test = df_test[['timestamp', 'lon', 'lat']].values
    y_test = df_test['P1'].values
    mu, var = model.predict_f(x_test)
    mse = mean_squared_error(y_test, mu[:, 0])
    print(mse)

    step = 662
    x_test = df_test[df_test['timestamp'] == step][[
        'timestamp', 'lon', 'lat'
        ]].values
    y_test = df_test[df_test['timestamp'] == step]['P1'].values
    lon_min, lon_max = x_test[:, 1].min(), x_test[:, 1].max()
    lat_min, lat_max = x_test[:, 2].min(), x_test[:, 2].max()

    new_lon, new_lat = np.mgrid[lon_min:lon_max:0.025, lat_min:lat_max:0.0125]
    xx = np.vstack((step * np.ones(new_lon.flatten().shape[0]),
                    new_lon.flatten(),
                    new_lat.flatten())).T
    print(xx)
    mx, _ = model.predict_f(xx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.hexbin(xx[:, 1], xx[:, 2], mx,
               gridsize=30, cmap='rainbow', alpha=0.7)
    plot = ax1.scatter(x_test[:, 1], x_test[:, 2],
                       c=y_test,
                       s=300, cmap='rainbow')
    ax2.scatter(x_test[:, 1], x_test[:, 2],
                c=y_test,
                s=300, cmap='rainbow')
    ax1.set_title("Data + Posterior")
    ax2.set_title("Data")
    plt.colorbar(plot)
    plt.show()
