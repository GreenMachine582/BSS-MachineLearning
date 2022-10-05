from __future__ import annotations

import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import Series
from sklearn.metrics import explained_variance_score, mean_squared_log_error, r2_score, mean_absolute_error,\
    mean_squared_error

# Constants
local_dir = os.path.dirname(__file__)


def plotPredictions(y_train: Series, y_test: Series, y_pred: ndarray) -> None:
    """
    Plot a line graph of BSS Demand and Predicted Demand.

    :param y_train: Training independent features, should be a Series
    :param y_test: Testing dependent features, should be a Series
    :param y_pred: Predicted dependent variables, should be a ndarray
    :return: None
    """
    temp_y_train = y_train.resample('D').sum()
    temp_y_test = y_test.resample('D').sum()
    temp_y_pred = Series(y_pred, index=y_test.index).resample('D').sum()

    plt.figure()
    plt.plot(temp_y_train.index, temp_y_train, c='b', label='Train')
    plt.plot(temp_y_test.index, temp_y_test, c='r', label='Test')
    plt.plot(temp_y_pred.index, temp_y_pred, c='g', label=f"Predictions")
    plt.title('BSS Predicted Demand')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()

    plt.figure()
    plt.plot(temp_y_test.index, temp_y_test['cnt'], c='r', label='Test')
    plt.plot(temp_y_pred.index, temp_y_pred, c='g', label=f"Predictions")
    plt.title('BSS Predicted Demand (Closeup)')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.show()


def resultAnalysis(y_test: Series, y_pred: ndarray, show: bool = True) -> dict:
    """
    Calculate and display the result analysis for estimators.

    :param y_test: Testing dependent variables, should be a Series
    :param y_pred: Model predictions, should be a ndarray
    :param show: Whether to show the results, should be a bool
    :return: results - dict[str: float]
    """
    logging.info("Analysing results")

    results = {'explained_variance': explained_variance_score(y_test, y_pred),
               'mean_squared_log_error': mean_squared_log_error(y_test, y_pred),
               'r2': r2_score(y_test, y_pred),
               'mae': mean_absolute_error(y_test, y_pred),
               'mse': mean_squared_error(y_test, y_pred)}
    results['rmse'] = np.sqrt(results['mse'])

    if show:
        print('explained_variance: %.4f' % results['explained_variance'])
        print('mean_squared_log_error: %.4f' % results['mean_squared_log_error'])
        print('r2: %.4f' % results['r2'])
        print('MAE: %.4f' % results['mae'])
        print('MSE: %.4f' % results['mse'])
        print('RMSE: %.4f' % np.sqrt(results['rmse']))
    return results
