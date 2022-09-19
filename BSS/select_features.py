from __future__ import annotations

import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

import BSS

# Constants
local_dir = os.path.dirname(__file__)


def trainModel(x_train, y_train):
    model = GradientBoostingRegressor()
    tscv = TimeSeriesSplit(n_splits=10)  # TimeSeries Cross validation
    cv_results = cross_val_score(model, x_train, y_train, cv=tscv, scoring='r2')
    return cv_results


def main(dir_=local_dir):
    config = BSS.Config(dir_, name='Bike-Sharing-Dataset-day', suffix='-pre-processed')

    dataset = BSS.Dataset(config)
    # Checks if BSS dataset was loaded
    if dataset.df is None:
        logging.warning(f"DataFrame object was expected, got '{type(dataset.df)}'")
        return

    # Process the dataset
    dataset.apply(BSS.process.processData)
    dataset.update(suffix='-processed')

    # edit these values
    num_of_runs = 100
    unselected_features = [['season'], ['yr'], ['mnth'], ['is_holiday'], ['weather_code'], ['temp'], ['atemp'],
                           ['hum'], ['wind_speed'], ['is_weekend'], ['prev'], ['diff'], ['prev-2'], ['diff-2'],
                           ['season', 'mnth'], ['season', 'yr'], ['yr', 'mnth'], ['season', 'yr', 'mnth']]

    unselected_features = [['season'], ['yr'], ['mnth'],
                           ['season', 'mnth'], ['season', 'yr'], ['yr', 'mnth'],
                           ['season', 'yr', 'mnth']]

    df = dataset.df.copy()
    results = []
    for features in unselected_features:
        dataset.update(df=df.drop(features, axis=1))
        dataset.handleMissingData()
        x, y = dataset.split()

        scores = np.empty(0)
        for i in range(num_of_runs):
            scores = np.concatenate((scores, trainModel(x['train'], y['train'])))
        results.append(scores)

    for features, result in zip(unselected_features, results):
        print(f"{features} - Mean: {result.mean()}, Std: {result.std()}")

    plt.figure()
    plt.boxplot(results, labels=unselected_features)
    plt.title('Unselected Feature Comparison')
    plt.show()


if __name__ == '__main__':
    main()
