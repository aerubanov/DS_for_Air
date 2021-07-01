import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf

from src.models.osgpr import OSGPR


data_file = 'DATA/processed/dataset.csv'
x_col = ['timestamp', 'lon', 'lat']
y_col = 'P1'

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
kernel = spat_cov * time_cov


def convert_time(data) -> pd.DataFrame:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = (
            data['timestamp'] - pd.to_datetime('2021-01-01', utc=True)
            )/pd.Timedelta(hours=1)
    return data


def time_cv(data: pd.DataFrame, step=24):
    curr_t = data['timestamp'].min()
    next_t = curr_t + step
    while curr_t < data['timestamp'].max():
        item = data[
                (data['timestamp'] >= curr_t) & (data['timestamp'] < next_t)
                ]
        if len(item) > 0:
            yield item
        curr_t = next_t
        next_t += step


def get_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    data.spat.set_y_col(y_col)
    qt = QuantileTransformer(
            output_distribution='normal',
            random_state=42,
            n_quantiles=100,
            )
    qt.fit(data.spat.y)

    data = data.spat.tloc['2021-01-01':'2021-02-01']
    data = data.dropna(subset=['P1'])
    data = data[['timestamp', 'lat', 'lon', 'P1', 'sds_sensor']]

    data.spat.y = qt.transform(data.spat.y.values).flatten()
    data.spat.x = convert_time(data)

    return data


def train_model(
        data: pd.DataFrame,
        M=200,
        max_iter=100
        ) -> gpflow.models.sgpr.SGPR:
    x = data[x_col].values
    y = data[y_col].values[:, None]

    Z = x[np.random.permutation(x.shape[0])[0:M], :]
    model = gpflow.models.sgpr.SGPR((x, y), kernel, Z)
    gpflow.set_trainable(model.kernel, False)

    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
            model.training_loss,
            model.trainable_variables,
            options={'iprint': 50, 'maxiter': max_iter},
            )
    return model


def update_model(
        model: gpflow.models.GPModel,
        data: pd.DataFrame,
        new_m: int = 20,
        max_iter: int = 1000,
        iprint: int = 50,
        ) -> gpflow.models.GPModel:
    x = data[x_col].values
    y = data[y_col].values[:, None]

    Z_opt = model.inducing_variable.Z
    mu, Su = model.predict_f(Z_opt, full_cov=True)
    if len(Su.shape) == 3:
        Su = Su[0, :, :]
    Kaa1 = model.kernel.K(model.inducing_variable.Z)

    Zinit = x[np.random.permutation(x.shape[0])[0:new_m], :]
    Zinit = np.vstack((Z_opt.numpy(), Zinit))

    new_model = OSGPR(
            (x, y),
            kernel,
            mu[new_m:, :],
            Su[new_m:, new_m:],
            Kaa1[new_m:, new_m:],
            Z_opt[new_m:, :],
            Zinit,
            )
    new_model.likelihood.variance.assign(model.likelihood.variance)
    for i, item in enumerate(model.kernel.trainable_variables):
        new_model.kernel.trainable_variables[i].assign(item)
    gpflow.set_trainable(model.kernel, False)
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
            new_model.training_loss,
            new_model.trainable_variables,
            options={'iprint': iprint, 'maxiter': max_iter},
            )
    return new_model


def eval_model(model: gpflow.models.GPModel, test_data):
    x_test = test_data[x_col].values
    y_test = test_data[y_col].values[:, None]

    mu, var = model.predict_f(x_test)
    mse = mean_squared_error(y_test, mu[:, 0])
    return mse


def plot_time(model: gpflow.models.GPModel, train_data, test_data):
    x_train = train_data[x_col].values
    x_test = test_data[x_col].values

    _, (ax1, ax2) = plt.subplots(1, 2,
                                 figsize=(15, 5),
                                 gridspec_kw={'width_ratios': [3, 1]})

    mu, var = model.predict_f(x_test)
    test_data.loc[:, 'prediction'] = mu[:, 0]
    df = test_data.groupby('timestamp').median().reset_index()
    df.plot(y=['P1', 'prediction'], ax=ax2)

    mu, _ = model.predict_f(x_train)
    train_data.loc[:, 'prediction'] = mu[:, 0]
    df = train_data.groupby('timestamp').median().reset_index()
    df.plot(y=['P1', 'prediction'], ax=ax1)

    plt.show()


def plot_spatial(model: gpflow.models.GPModel, data, timestamp=None):
    if timestamp is None:
        timestamp = train_data['timestamp'].max()
    df = data[train_data['timestamp'] == timestamp]
    x = df[x_col].values
    y = df[y_col].values

    lon_min, lon_max = x[:, 1].min(), x[:, 1].max()
    lat_min, lat_max = x[:, 2].min(), x[:, 2].max()

    new_lon, new_lat = np.mgrid[lon_min:lon_max:0.025, lat_min:lat_max:0.0125]
    xx = np.vstack((x[0, 0] * np.ones(new_lon.flatten().shape[0]),
                    new_lon.flatten(),
                    new_lat.flatten())).T
    mx, var = model.predict_f(xx)

    f, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))
    plot1 = ax3.hexbin(xx[:, 1], xx[:, 2], mx,
                       gridsize=30, cmap='rainbow', alpha=0.7)
    ax3.scatter(x[:, 1], x[:, 2],
                c=y,
                s=300, cmap='rainbow')
    plot2 = ax4.hexbin(xx[:, 1], xx[:, 2], np.sqrt(var),
                       gridsize=30, cmap='rainbow', alpha=0.7)
    plt.colorbar(plot1, ax=ax3)
    plt.colorbar(plot2, ax=ax4)
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)
    data = get_data(data_file)
    train_t = 13*24
    test_t = 13*24 + 24
    results = []
    train_data = data[data['timestamp'] < train_t]
    test_data = data[
            (data['timestamp'] >= train_t) & (data['timestamp'] < test_t)
            ]
    model = train_model(train_data, max_iter=100)
    mse = eval_model(model, test_data)
    print(f'init MSE: {mse}')
    results.append(mse)

    for i, item in enumerate(time_cv(data[data['timestamp'] >= test_t])):
        train_data = test_data
        test_data = item
        model = update_model(model, train_data, max_iter=100, iprint=0)
        mse = eval_model(model, test_data)
        print(f'step {i} MSE: {mse}')
        results.append(mse)
    print(np.mean(np.sqrt(results)), np.std(np.sqrt(results)))
    _, ax = plt.subplots(figsize=(15, 5))
    ax.plot([i for i in range(len(results))], np.sqrt(results))
    ax.set_xlabel('iteration')
    ax.set_ylabel('RMSE')
    plt.show()
    plot_spatial(model, train_data)
