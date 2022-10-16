import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from mlxtend.evaluate import bias_variance_decomp
from sklearn.ensemble import GradientBoostingRegressor

import BSS
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def _plotBullseye(ax, x, y, title: str = ''):
    circle1 = plt.Circle((0, 0), 0.08, color='red', alpha=0.6)
    circle2 = plt.Circle((0, 0), 0.25, color='blue', linewidth=0.8, alpha=0.3, fill=False)
    circle3 = plt.Circle((0, 0), 0.55, color='red', linewidth=0.8, alpha=0.3, fill=False)
    circle4 = plt.Circle((0, 0), 0.85, color='blue', linewidth=0.8, alpha=0.3, fill=False)
    ax.add_artist(circle4)
    ax.add_artist(circle3)
    ax.add_artist(circle2)
    ax.add_artist(circle1)

    ax.set_aspect('equal')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(title)

    ax.scatter(x, y, c='g')
    ax.axis([-1, 1, -1, 1])
    return ax


def biasVarianceDecomp(X_train, X_test, y_train, y_test, n_iter: int = 10, display: bool = False,
                       dataset_name: str = '', dir_: str = ''):
    estimator = BSS.compare_params.getMLPRegressor()
    model = estimator['base'].set_params(**estimator['best_params'])
    loss, bias, var = [], [], []
    for i in range(n_iter):
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, X_train.values, y_train.values,
                                                                    X_test.values, y_test.values,
                                                                    num_rounds=5, loss='mse')
        if display:
            print('Average expected loss: %.3f' % avg_expected_loss)
            print('Average bias: %.3f' % avg_bias)
            print('Average variance: %.3f' % avg_var)
        loss.append(avg_expected_loss)
        bias.append(avg_bias)
        var.append(avg_var)
        logging.info(f"Bias variance iter: {i + 1}/{n_iter}")

    fig, ax = plt.subplots()
    _plotBullseye(ax, bias, var, title=estimator['fullname'])
    plt.show()


def main(dir_=local_dir):
    config = ml.Config(dir_, 'DC_day')

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        raise Exception("Failed to load dataset")
    dataset = BSS.processDataset(dataset)

    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    best_params = {'criterion': 'squared_error',
                   'learning_rate': 0.078,
                   'max_depth': 2,
                   'n_estimators': 1050,
                   'subsample': 0.3}

    loss, bias, var = [], [], []
    for _ in range(10):
        avg_expected_loss, avg_bias, avg_var = biasVarianceDecomp(GradientBoostingRegressor(**best_params),
                                                                  X_train, X_test, y_train, y_test,
                                                                  n_iter=2)
        # print('Average expected loss: %.3f' % avg_expected_loss)
        # print('Average bias: %.3f' % avg_bias)
        # print('Average variance: %.3f' % avg_var)
        loss.append(avg_expected_loss)
        bias.append(avg_bias)
        var.append(avg_var)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    _plotBullseye(ax1, bias[0], var[0])
    # _plotBullseye(ax2, bias[1], var[1], names[1])
    # _plotBullseye(ax3, bias[2], var[2], names[2])
    # _plotBullseye(ax4, bias[3], var[3], names[3])

    plt.show()

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
