from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from sklearn import metrics

from machine_learning import utils


def plotPrediction(y_train: Series, y_test: Series, y_pred: tuple | dict | list, target: str = 'Target',
                   dataset_name: str = '', dir_: str = '') -> None:
    """
    Plot the prediction on a line graph.

    :param y_train: Training independent features, should be a Series
    :param y_test: Testing dependent features, should be a Series
    :param y_pred: Predicted dependent variables, should be a tuple[str, ndarray] | dict[str: ndarray
     | list[tuple[str, ndarray]
    :param target: The predicted variables name, should be a str
    :param dataset_name: Name of dataset, should be a str
    :param dir_: Save location for figures, should be a str
    :return: None
    """
    logging.info("Plotting predictions")

    if isinstance(y_pred, tuple):
        y_preds = [y_pred]
    elif isinstance(y_pred, dict):
        y_preds = [(x, y_pred[x]) for x in y_pred]
    elif isinstance(y_pred, list):
        y_preds = y_pred
    else:
        raise TypeError(f"'y_pred': Expected type 'tuple | dict | list', got {type(y_pred).__name__} instead")

    # plots a line graph of BSS True and Predicted demand
    fig, ax = plt.subplots(figsize=(10, 5))
    colour = iter(plt.cm.rainbow(np.linspace(0, 1, len(y_preds) + 2)))
    plt.plot(y_train.index, y_train, c=next(colour), label='Train')
    plt.plot(y_test.index, y_test, c=next(colour), label='Test')
    for name, y_pred in y_preds:
        plt.plot(y_test.index, y_pred, c=next(colour), label=f"{name} Predictions")
    plt.legend()
    ax.set(xlabel='Date', ylabel=target)
    fig.suptitle(f"Estimator Prediction - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))

    # plots a closeup view if the test data and predictions
    fig, ax = plt.subplots(figsize=(16, 6))
    colour = iter(plt.cm.rainbow(np.linspace(0, 1, len(y_preds) + 2)))
    plt.plot(y_test.index, y_test, c=next(colour), label='Test')
    for name, y_pred in y_preds:
        plt.plot(y_test.index, y_pred, c=next(colour), label=f"{name} Predictions")
    plt.legend()
    ax.set(xlabel='Date', ylabel=target)
    fig.suptitle(f"Estimator Prediction (Closeup) - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
    plt.show()


def resultAnalysis(y_test: Series, y_pred: tuple | dict | list, plot: bool = True, display: bool = True,
                   dataset_name: str = '', dir_: str = '') -> dict:
    """
    Calculate the result analysis with options to display and plot.

    :param y_test: Testing dependent variables, should be a Series
    :param y_pred: Predicted dependent variables, should be a tuple[str, ndarray] | dict[str: ndarray
     | list[tuple[str, ndarray]
    :param plot: Whether to plot the results, should be a bool
    :param display: Whether to display the results, should be a bool
    :param dataset_name: Name of dataset, should be a str
    :param dir_: Save location for figures, should be a str
    :return: results - dict[str: list[str | float]]
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
        results['explained_variance'].append(metrics.explained_variance_score(y_test, y_pred))
        results['mean_squared_log_error'].append(metrics.mean_squared_log_error(y_test, y_pred))
        results['r2'].append(metrics.r2_score(y_test, y_pred))
        results['mae'].append(metrics.mean_absolute_error(y_test, y_pred))
        results['mse'].append(metrics.mean_squared_error(y_test, y_pred))
        results['rmse'].append(np.sqrt(results['mse'][-1]))

        if display:
            print("\nModel:", name)
            print("Explained variance: %.4f" % results['explained_variance'][-1])
            print("Mean Squared Log Error: %.4f" % results['mean_squared_log_error'][-1])
            print("R2: %.4f" % results['r2'][-1])
            print("MAE: %.4f" % results['mae'][-1])
            print("MSE: %.4f" % results['mse'][-1])
            print("RMSE: %.4f" % results['rmse'][-1])

    if not plot:
        return results

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
    return results
