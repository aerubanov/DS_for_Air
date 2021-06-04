import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import gpflow


data_file = 'DATA/processed/dataset.csv'
x_col = ['timestamp', 'lon', 'lat']
y_col = ['P1']

def convert_time(data) -> pd.DataFrame:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = (
            data['timestamp'] - pd.to_datetime('2021-01-01', utc=True)
            )/pd.Timedelta(hours=1)
    return data


def get_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    data.spat.set_y_col(y_col)
    qt = QuantileTransformer(
            output_distribution='normal',
            random_state=42,
            n_quantiles=100,
            )
    qt.fit(data.spat.y)

    data = data.spat.tloc['2021-01-01':'2021-04-01']
    data = data.dropna(subset=['P1'])
    data = data[['timestamp', 'lat', 'lon', 'P1', 'sds_sensor']]

    data.spat.y = qt.transform(data.spat.y.values).flatten()
    data.spat.x = convert_time(data)

    return data


def train_model(data: pd.DataFrame, M=200, max_iter=100) -> gpflow.models.sgpr.SGPR:
    x = data[x_col].values
    y = data[y_col].values[:, None]

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
            options={'iprint': 50, 'maxiter': max_iter},
            )
    return model


def update_model(
        model: gpflow.models.GPModel,
        data: pd.DataFrame,
        new_m: int = 20,
        ) -> gpflow.models.GPModel:
    x = data[x_col].values
    y = data[y_col].values[:, None]

    Z_opt = model.inducing_variable.Z
    mu, Su = model.predict_f(Z_opt, full_cov=True)
    if len(Su.shape) == 3:
        Su = Su[0, :, :]
    Kaa1 = model.kernel.K(model.inducing_variable.Z)

    Zinit = x_new[np.random.permutation(x_new.shape[0])[0:new_m], :]
    Zinit = np.vstack((Z_opt.numpy(), Zinit))

    new_model = OSGPR(
            (x_new, y_new),
            kernel,
            mu,
            Su,
            Kaa1,
            Z_opt,
            Zinit,
            )
    new_model.likelihood.variance.assign(model.likelihood.variance)
    for i, item in enumerate(model.kernel.trainable_variables):
        new_model.kernel.trainable_variables[i].assign(item)
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(new_model.training_loss, new_model.trainable_variables)
    return new_model


def eval_model(model: gpflow.models.GPModel, train_data, test_data):
    x_train = train_data[x_col].values
    x_test = test_data[x_col].values
    y_test = test_data[y_col].values[:, None]

    _, (ax1, ax2) = plt.subplots(1, 2,
                                 figsize=(15, 5),
                                 gridspec_kw={'width_ratios': [3, 1]})

    mu, var = model.predict_f(x_test)
    mse = mean_squared_error(y_test, mu[:, 0])
    test_data.loc[:, 'prediction'] = mu[:, 0]
    df = test_data.groupby('timestamp').median().reset_index()
    df.plot(y=['P1', 'prediction'], ax=ax2)

    mu, _ = model.predict_f(x_train)
    train_data.loc[:, 'prediction'] = mu[:, 0]
    df = train_data.groupby('timestamp').median().reset_index()
    df.plot(y=['P1', 'prediction'], ax=ax1)

    df = train_data[train_data['timestamp'] == train_data['timestamp'].max()]
    x1 = df[x_col].values
    y1 = df[y_col].values

    lon_min, lon_max = x1[:, 1].min(), x1[:, 1].max()
    lat_min, lat_max = x1[:, 2].min(), x1[:, 2].max()

    new_lon, new_lat = np.mgrid[lon_min:lon_max:0.025, lat_min:lat_max:0.0125]
    xx = np.vstack((x1[0, 0] * np.ones(new_lon.flatten().shape[0]),
                    new_lon.flatten(),
                    new_lat.flatten())).T
    mx, var = model.predict_f(xx)

    f, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))
    plot1 = ax3.hexbin(xx[:, 1], xx[:, 2], mx,
                       gridsize=30, cmap='rainbow', alpha=0.7)
    ax3.scatter(x1[:, 1], x1[:, 2],
                c=y1,
                s=300, cmap='rainbow')
    plot2 = ax4.hexbin(xx[:, 1], xx[:, 2], np.sqrt(var),
                       gridsize=30, cmap='rainbow', alpha=0.7)
    plt.colorbar(plot1, ax=ax3)
    plt.colorbar(plot2, ax=ax4)
    print(f'MSE: {mse}')
    plt.show()
    return mse


if __name__ == '__main__':
    data = get_data(data_file)
    print(data['timestamp'].min(), data['timestamp'].max())
    train_data = data[data['timestamp'] < 24 * 13]
    test_data = data[
            (data['timestamp'] >= 24 * 13) & (data['timestamp'] < 24 * 14)
            ]
    print(len(train_data), len(test_data))
    model = train_model(train_data, max_iter=300)
    _ = eval_model(model, train_data, test_data)
