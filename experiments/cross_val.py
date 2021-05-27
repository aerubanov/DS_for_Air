from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.models.osgpr import OSGPR

data_file = 'DATA/processed/dataset.csv'

mt = gpflow.kernels.Matern32(variance=1, lengthscales=24)
pk = gpflow.kernels.Periodic(
        gpflow.kernels.SquaredExponential(lengthscales=24*1.5),
        period=24,
        )
kernel = mt * pk


def get_data(file_path: str) -> (np.ndarray, np.ndarray):
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

    x = np.arange(len(data)).astype(np.float64)
    y = data.spat.y.values
    y = qt.transform(y).flatten()
    return x, y


def plot_result(model, cur_x, cur_y,  x_star, y_star):
    _, ax = plt.subplots(figsize=(15, 5))
    xx = np.linspace(cur_x.min(), x_star.max(), 200)[:, None]
    mu, var = model.predict_f(xx)
    Zopt = model.inducing_variable.Z
    ax.plot(cur_x, cur_y, 'kx', mew=1, alpha=0.8)
    ax.plot(x_star, y_star, 'kx', mew=1, alpha=0.8)
    ax.plot(xx, mu, 'b', lw=2)
    ax.fill_between(
        xx[:, 0],
        mu[:, 0] - 2 * np.sqrt(var),
        mu[:, 0] + 2 * np.sqrt(var),
        color='b', alpha=0.3)

    mu, _ = model.predict_f(Zopt)
    ax.plot(Zopt.numpy(), mu.numpy(), 'ro', mew=1)
    y_pr, _ = model.predict_f(x_star)
    rmse = mean_squared_error(y_star, y_pr[:, 0])
    print(rmse)
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False,
                      edgecolor='none', linewidth=0)
    ax.legend([extra], (f"RMSE: {rmse}",))
    plt.savefig("experiments/plots/SGPR_time.png")
    plt.show()


def train_init_model(
        X: np.ndarray,
        y: np.ndarray,
        m: int = 50,
        ) -> gpflow.models.GPModel:
    X, y = X[:, None], y[:, None]
    X = X.astype(np.float64)

    Z = X[np.random.permutation(X.shape[0])[0:m], :]

    model = gpflow.models.sgpr.SGPR((X, y), kernel, Z)

    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(model.training_loss, model.trainable_variables)
    return model


def update_model(
        model: gpflow.models.GPModel,
        x_new: np.ndarray,
        y_new: np.ndarray,
        new_m: int = 5
        ) -> gpflow.models.GPModel:
    Z_opt = model.inducing_variable.Z
    mu, Su = model.predict_f(Z_opt, full_cov=True)
    if len(Su.shape) == 3:
        Su = Su[0, :, :]
    Kaa1 = model.kernel.K(model.inducing_variable.Z)

    x_new, y_new = x_new[:, None], y_new[:, None]
    x_new = x_new.astype(np.float64)
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


if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)

    x, y = get_data(data_file)
    n_splits = round((len(x)-20*24)/24)
    cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=20*24, test_size=24)
    splits = cv.split(x)

    train_index, test_index = next(splits)
    model = train_init_model(X=x[train_index],
                             y=y[train_index],
                             m=100,
                             )
    y_pred, _ = model.predict_f(x[test_index][:, None].astype(np.float64))
    rmse = mean_squared_error(
            y[test_index],
            y_pred)
    print(f'init MSE: {rmse}')

    x_new = x[test_index]
    y_new = y[test_index]
    i = 1
    scores = []
    for train_index, test_index in splits:
        model = update_model(model, x_new, y_new)
        y_pred, _ = model.predict_f(x[test_index][:, None].astype(np.float64))
        rmse = mean_squared_error(
                y[test_index],
                y_pred[:, 0])
        print(f'iteration {i} MSE: {rmse}')
        scores.append(rmse)
        x_new = x[test_index]
        y_new = y[test_index]
        i += 1

    print(f'MSE: {np.mean(scores)} STD: {np.std(scores)}')
    plot_result(
            model,
            x[:len(x)-24][:, None],
            y[:len(x)-24][:, None],
            x[len(x)-24:][:, None],
            y[len(x)-24:][:, None],
        )
