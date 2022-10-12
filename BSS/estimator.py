from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace
from pandas import Series
from sklearn.metrics import explained_variance_score, mean_squared_log_error, r2_score, mean_absolute_error, \
    mean_squared_error

from machine_learning import utils


def plotPredictions(y_train: Series, y_test: Series, y_pred: tuple | dict | list, dataset_name: str = '',
                    dir_: str = '') -> None:
    """
    Plot a line graph of BSS Demand and Predicted Demand.

    :param y_train: Training independent features, should be a Series
    :param y_test: Testing dependent features, should be a Series
    :param y_pred: Predicted dependent variables, should be a tuple[str, ndarray] | dict[str: ndarray
     | list[tuple[str, ndarray]
    :param dataset_name: Name of dataset, should be a str
    :param dir_: Save location for figures, should be a str
    :return: None
    """
    logging.info("Plotting predictions")
    temp_y_train = y_train.resample('D').sum()
    temp_y_test = y_test.resample('D').sum()

    if isinstance(y_pred, tuple):
        y_preds = [(y_pred[0], Series(y_pred[1], index=y_test.index).resample('D').sum())]
    elif isinstance(y_pred, dict):
        y_preds = [(x, Series(y_pred[x], index=y_test.index).resample('D').sum()) for x in y_pred]
    elif isinstance(y_pred, list):
        y_preds = [(x[0], Series(x[1], index=y_test.index).resample('D').sum()) for x in y_pred]
    else:
        raise TypeError(f"'y_pred': Expected type 'tuple | dict | list', got {type(y_pred).__name__} instead")

    # plots a line graph of BSS True and Predicted demand
    fig, ax = plt.subplots(figsize=(10, 5))
    colour = iter(plt.cm.rainbow(linspace(0, 1, len(y_preds) + 2)))
    plt.plot(temp_y_train.index, temp_y_train, c=next(colour), label='Train')
    plt.plot(temp_y_test.index, temp_y_test, c=next(colour), label='Test')
    for name, y_pred in y_preds:
        plt.plot(temp_y_test.index, y_pred, c=next(colour), label=f"{name} Predictions")
    plt.legend()
    ax.set(xlabel='Date', ylabel='Demand')
    fig.suptitle(f"BSS Predicted Demand - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))

    # plots a closeup view if the test data and predictions
    fig, ax = plt.subplots(figsize=(16, 6))
    colour = iter(plt.cm.rainbow(linspace(0, 1, len(y_preds) + 2)))
    plt.plot(temp_y_test.index, temp_y_test, c=next(colour), label='Test')
    for name, y_pred in y_preds:
        plt.plot(temp_y_test.index, y_pred, c=next(colour), label=f"{name} Predictions")
    plt.legend()
    ax.set(xlabel='Date', ylabel='Demand')
    fig.suptitle(f"BSS Predicted Demand (Closeup) - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
    plt.show()


def plotResultAnalysis(y_test: Series, y_pred: tuple | dict | list, show: bool = True, dataset_name: str = '',
                       dir_: str = '') -> None:
    """
    Calculate and display the result analysis for estimators.

    :param y_test: Testing dependent variables, should be a Series
    :param y_pred: Predicted dependent variables, should be a tuple[str, ndarray] | dict[str: ndarray
     | list[tuple[str, ndarray]
    :param show: Whether to show the results, should be a bool
    :param dataset_name: Name of dataset, should be a str
    :param dir_: Save location for figures, should be a str
    :return: None
    """
    logging.info("Analysing results")

    if isinstance(y_pred, tuple):
        y_preds = [y_pred]
    elif isinstance(y_pred, dict):
        y_preds = [(x, y_pred[x]) for x in y_pred]
    elif isinstance(y_pred, list):
        y_preds = y_pred
    else:
        raise TypeError(f"'y_pred': Expected type 'tuple | dict | list', got {type(y_pred).__name__} instead")

    results = {'names': [], 'explained_variance': [], 'mean_squared_log_error': [],
               'r2': [], 'mae': [], 'mse': [], 'rmse': []}

    for name, y_pred in y_preds:
        results['names'].append(name)
        results['explained_variance'].append(explained_variance_score(y_test, y_pred))
        results['mean_squared_log_error'].append(mean_squared_log_error(y_test, y_pred))
        results['r2'].append(r2_score(y_test, y_pred))
        results['mae'].append(mean_absolute_error(y_test, y_pred))
        results['mse'].append(mean_squared_error(y_test, y_pred))
        results['rmse'].append(np.sqrt(results['mse'][-1]))

        if show:
            print("\nModel:", name)
            print("Explained variance: %.4f" % results['explained_variance'][-1])
            print("Mean Squared Log Error: %.4f" % results['mean_squared_log_error'][-1])
            print("R2: %.4f" % results['r2'][-1])
            print("MAE: %.4f" % results['mae'][-1])
            print("MSE: %.4f" % results['mse'][-1])
            print("RMSE: %.4f" % results['rmse'][-1])

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8), sharex='col')
    utils._plotBar(ax1, results['names'], results['explained_variance'], 'Explained Variance')
    utils._plotBar(ax2, results['names'], results['mean_squared_log_error'], 'Mean Squared Log Error')
    utils._plotBar(ax3, results['names'], results['r2'], 'R2')
    utils._plotBar(ax4, results['names'], results['mae'], 'MAE')
    utils._plotBar(ax5, results['names'], results['mse'], 'MSE')
    utils._plotBar(ax6, results['names'], results['rmse'], 'RMSE')
    fig.suptitle(f"Result Analysis - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
    plt.show()
