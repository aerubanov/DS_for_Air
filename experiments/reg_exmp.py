import numpy as np
import gpflow as GPflow
import src.models.osgpr
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.ticker as ticker


def figsize(scale, ratio=None):
    fig_width_pt = 397.4849  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 4  # Aesthetic ratio (you could change this)
    if ratio is not None:
        golden_mean = ratio
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def init_Z(cur_Z, new_X, use_old_Z=True, first_batch=True):
    if use_old_Z:
        Z = np.copy(cur_Z)
    else:
        M = cur_Z.shape[0]
        M_old = int(0.7 * M)
        M_new = M - M_old
        old_Z = cur_Z[np.random.permutation(M)[0:M_old], :]
        new_Z = new_X[np.random.permutation(new_X.shape[0])[0:M_new], :]
        Z = np.vstack((old_Z, new_Z))
    return Z


def plot_model(model, ax, cur_x, cur_y, pred_x, seen_x=None, seen_y=None):
    mx, vx = model.predict_f(pred_x)
    Zopt = model.inducing_variable.Z
    mu, Su = model.predict_f(Zopt, full_cov=True)
    if len(Su.shape) == 3:
        Su = Su[0, :, :]
        vx = vx[:, 0]
    ax.plot(cur_x, cur_y, 'kx', mew=1, alpha=0.8)
    if seen_x is not None:
        ax.plot(seen_x, seen_y, 'kx', mew=1, alpha=0.2)
    ax.plot(pred_x, mx, 'b', lw=2)
    ax.fill_between(
        pred_x[:, 0],
        mx[:, 0] - 2 * np.sqrt(vx),
        mx[:, 0] + 2 * np.sqrt(vx),
        color='b', alpha=0.3)
    ax.plot(Zopt.numpy(), mu.numpy(), 'ro', mew=1)
    ax.set_ylim([-2.4, 2])
    ax.set_xlim([np.min(pred_x), np.max(pred_x)])
    plt.subplots_adjust(hspace=.08)
    ax.set_ylabel('y')
    ax.yaxis.set_ticks(np.arange(-2, 3, 1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    return mu, Su, Zopt


def get_data(shuffle):
    X = np.loadtxt('experiments/reg_toy_x.txt', delimiter=',')
    y = np.loadtxt('experiments/reg_toy_y.txt', delimiter=',')

    X = X.reshape(X.shape[0], 1)
    y = y.reshape((y.shape[0], 1))
    N = y.shape[0]
    gap = round(N / 3)
    X[:gap, :] = X[:gap, :] - 1
    X[2 * gap:3 * gap, :] = X[2 * gap:3 * gap, :] + 1
    if shuffle:
        idxs = np.random.permutation(N)
        X = X[idxs, :]
        y = y[idxs, :]
    return X, y


def test_data(n=100):
    np.random.seed(1)
    x = np.linspace(0, 5, n)
    y = np.sin(x) + np.random.normal(loc=0, scale=0.2, size=n)
    # plt.plot(x, y)
    # plt.show()
    x = x.reshape(x.shape[0], 1)
    y = y.reshape((y.shape[0], 1))
    return x, y


def plot_VFE_optimized(M, use_old_Z, shuffle):
    fig, axs = plt.subplots(4, 1, figsize=figsize(1, ratio=12.0 / 19.0), sharey=True, sharex=True)

    X, y = get_data(shuffle)
    # X, y = test_data(100)

    N = X.shape[0]
    gap = round(N / 3)

    # get the first portion and call sparse GP regression
    X1 = X[:gap, :]
    y1 = y[:gap, :]
    seen_x = None
    seen_y = None
    Z1 = X1[np.random.permutation(X1.shape[0])[0:M], :]

    model1 = GPflow.models.sgpr.SGPR((X1, y1), GPflow.kernels.RBF(1), Z1)
    model1.likelihood.variance.assign(0.001)
    model1.kernel.variance.assign(1.0)
    model1.kernel.lengthscales.assign(0.8)
    optimizer = GPflow.optimizers.Scipy()
    optimizer.minimize(model1.training_loss, model1.trainable_variables)

    # plot prediction
    xx = np.linspace(-2, 12, 100)[:, None]
    mu1, Su1, Zopt = plot_model(model1, axs[0], X1, y1, xx, seen_x, seen_y)

    # now call online method on the second portion of the data
    X2 = X[gap:2 * gap, :]
    y2 = y[gap:2 * gap, :]
    seen_x = X[:gap, :]
    seen_y = y[:gap, :]

    Kaa1 = model1.kernel.K(model1.inducing_variable.Z)
    # Zinit = init_Z(Zopt.numpy(), X2, use_old_Z)
    Zinit = X2[np.random.permutation(X1.shape[0])[0:M], :]
    Zinit = np.vstack((Zopt.numpy(), Zinit))
    print(Su1.shape)
    model2 = src.models.osgpr.OSGPR((X2, y2), GPflow.kernels.RBF(1), mu1, Su1, Kaa1,
                                    Zopt, Zinit)
    model2.likelihood.variance.assign(model1.likelihood.variance)
    model2.kernel.variance.assign(model1.kernel.variance)
    model2.kernel.lengthscales.assign(model1.kernel.lengthscales)
    optimizer = GPflow.optimizers.Scipy()
    optimizer.minimize(model2.training_loss, model2.trainable_variables, compile=False)

    # plot prediction
    mu2, Su2, Zopt = plot_model(model2, axs[1], X2, y2, xx, seen_x, seen_y)

    # now call online method on the third portion of the data
    X3 = X[2 * gap:3 * gap, :]
    y3 = y[2 * gap:3 * gap, :]
    seen_x = np.vstack((seen_x, X2))
    seen_y = np.vstack((seen_y, y2))

    Kaa2 = model2.kernel.K(tf.constant(model2.inducing_variable.Z))

    # Zinit = init_Z(Zopt.numpy(), X3, use_old_Z)
    Zinit = X2[np.random.permutation(X1.shape[0])[0:M], :]
    Zinit = np.vstack((Zopt.numpy(), Zinit))
    print(Su2.shape)
    model3 = src.models.osgpr.OSGPR((X3, y3), GPflow.kernels.RBF(1), mu2[10:, :], Su2[10:, 10:], Kaa2[10:, 10:],
                                    Zopt[10:, :], Zinit)
    model3.likelihood.variance.assign(model2.likelihood.variance)
    model3.kernel.variance.assign(model2.kernel.variance)
    model3.kernel.lengthscales.assign(model2.kernel.lengthscales)
    optimizer = GPflow.optimizers.Scipy()
    optimizer.minimize(model3.training_loss, model3.trainable_variables)
    mu3, Su3, Zopt = plot_model(model3, axs[2], X3, y3, xx, seen_x, seen_y)

    Z4 = X[np.random.permutation(X.shape[0])[0:M], :]
    model4 = GPflow.models.sgpr.SGPR((X, y), GPflow.kernels.RBF(1), Z4)
    model4.likelihood.variance.assign(0.001)
    model4.kernel.variance.assign(1.0)
    model4.kernel.lengthscales.assign(0.8)
    optimizer = GPflow.optimizers.Scipy()
    optimizer.minimize(model4.training_loss, model4.trainable_variables)

    # plot prediction
    xx = np.linspace(-2, 12, 100)[:, None]
    mu4, Su4, Zopt4 = plot_model(model4, axs[3], X, y, xx, None, None)

    axs[3].set_xlabel('x')
    fig.savefig('experiments/plots/online_exmpl.png', bbox_inches='tight')


if __name__ == '__main__':
    use_old_Z = False
    alpha = 0.5
    seed = 10
    shuffle = False

    np.random.seed(seed)
    plot_VFE_optimized(10, use_old_Z, shuffle)
