import pandas as pd
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import arviz

from  src.experiments.data_gen import genetate_data


def get_data(file_path: str):
    data = pd.read_csv(file_path)
    data.spat.set_y_col('P1')
    data = data[data.sds_sensor == 56073]
    qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=100)
    # qt = PowerTransformer()
    qt.fit(data.spat.y)

    data = data.spat.tloc['2021-01-01':'2021-01-31']

    x = np.arange(len(data))[:, None]
    y = data.spat.y.values
    y = qt.transform(y)
    idx = round(len(data)*0.7)
    x, y, x_star, y_star = x[:idx], y[:idx], x[idx:], y[idx:]
    return x, y, x_star, y_star


if __name__ == '__main__':
    data_file = 'DATA/processed/dataset.csv'
    x, y, x_star, _ = get_data(data_file)
    # x, y, x_star, _ = genetate_data(100)

    with pm.Model() as model:
        # p = pm.Uniform('p', lower=5, upper=10)
        # l = pm.Uniform('l', lower=2, upper=7)
        # p = pm.Normal('p', mu=5, sigma=1)
        # l = pm.Normal('l', mu=5, sigma=1)
        # lm = pm.Normal('lm', mu=15, sigma=1)
        periods = [2.2, 3, 6.6, 9.7, 14]
        mt = pm.gp.cov.Matern32(1, ls=15)
        pk = pm.gp.cov.Periodic(1, period=3, ls=5)
        cov = mt + sum([pm.gp.cov.Matern32(1, ls=i * 2) * pm.gp.cov.Periodic(1, period=i, ls=1.5*i) for i in periods])

        gp = pm.gp.Latent(cov_func=cov)
        pr = gp.prior('pr', X=x)
        sigma = pm.HalfNormal('sigma', sigma=2)
        f_star = gp.conditional("f_star", x_star)
    with model:
        check = pm.sample_prior_predictive(samples=3)

    plt.plot(x.flatten(), y, label="data", color='red')
    for i in range(check['pr'].shape[0]):
        plt.plot(x.flatten(), check['pr'][i], alpha=0.3)
    plt.legend()
    plt.show()

    with model:
        y_ = pm.Normal("y", mu=pr, sigma=sigma, observed=y)

    with model:
        trace = pm.sample(200, n_init=100, tune=100, chains=2, cores=2, return_inferencedata=True)
        arviz.to_netcdf(trace, 'src/experiments/results/gp_time_trace')

    n_nonconverged = int(np.sum(arviz.rhat(trace)[["sigma", "pr_rotated_"]].to_array() > 1.03).values)
    print("%i variables MCMC chains appear not to have converged." % n_nonconverged)