from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

import BSS
from BSS.test import extractFeatures

# Constants
local_dir = os.path.dirname(__file__)


def trainModel(X_train: DataFrame, y_train: DataFrame, time_series: bool = False) -> ndarray:
    """
    Trains and cross validates a basic model with default params.

    :param X_train: training independent features, should be a DataFrame
    :param y_train: training dependent variables, should be a DataFrame
    :param time_series: whether the dataset is a time series, should be a bool
    :return: cv_results - ndarray[float]
    """
    model = GradientBoostingRegressor()
    cv = 10  # n Folds
    if time_series:
        cv = TimeSeriesSplit(n_splits=10)  # TimeSeries Cross validation
    cv_results = cross_val_score(model, X_train, y_train, cv=cv)
    logging.info("Model has been cross validated")
    return cv_results


def main(dir_: str = local_dir):
    """


    :param dir_: project's path directory, should be a str
    :return:
    """
    # TODO: Documentation and error handle
    name = 'london_merged-hour'
    config = BSS.Config(dir_, name)
    if not config.load():
        config.save()

    # Loads the BSS dataset
    dataset = BSS.Dataset(config.dataset, name=name + '-pre-processed')
    if not dataset.load():
        return

    # Process the dataset
    dataset.apply(BSS.processData)

    # edit these values
    num_of_runs = 5
    unselected_features = [[], ['yr'], ['is_holiday'], ['is_weekend'], ['season_1', 'season_2', 'season_3', 'season_4'],
                           ['mnth_1', 'mnth_2', 'mnth_3', 'mnth_4', 'mnth_5', 'mnth_6',
                            'mnth_7', 'mnth_8', 'mnth_9', 'mnth_10', 'mnth_11', 'mnth_12'],
                           ['season_1', 'season_2', 'season_3', 'season_4', 'mnth_1', 'mnth_2', 'mnth_3', 'mnth_4',
                            'mnth_5', 'mnth_6', 'mnth_7', 'mnth_8', 'mnth_9', 'mnth_10', 'mnth_11', 'mnth_12']]

    dataset.apply(extractFeatures, dataset.target)

    df = dataset.df.copy()
    results = []
    for features in unselected_features:
        dataset.update(df=df.drop(features, axis=1))
        dataset.apply(BSS.handleMissingData)

        X_train, X_test, y_train, y_test = dataset.split(config.model['random_seed'])

        scores = np.empty(0)
        threads = {}
        with ThreadPoolExecutor(max_workers=max(1, os.cpu_count()-2)) as executor:
            for thread_key in range(num_of_runs):
                threads[thread_key] = executor.submit(trainModel, *(X_train, y_train), time_series=True)

            for thread_key in threads:
                scores = np.append(scores, threads[thread_key].result())
        results.append(scores)

    for features, result in zip(unselected_features, results):
        print(f"{features} - Mean: {result.mean()}, Std: {result.std()}")

    plt.figure()
    plt.boxplot(results, labels=unselected_features)
    plt.title('Unselected Feature Comparison')
    plt.show()

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
