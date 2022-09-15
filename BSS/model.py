from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn import metrics

from BSS import utils, Config


def gridSearch(model, param_search, x_train, y_train, n_split=10, n_jobs=-1, verbose=2):
    logging.info("Grid Searching model hyperparameters")
    tscv = TimeSeriesSplit(n_splits=n_split)
    grid_search = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, n_jobs=n_jobs, verbose=verbose)
    grid_search.fit(x_train, y_train)
    return grid_search


def regressionResults(y_test, y_pred):
    # Regression metrics
    results = []
    explained_variance = metrics.explained_variance_score(y_test, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    results.append(('explained_variance:', round(explained_variance, 4)))
    results.append(('mean_squared_log_error:', round(mean_squared_log_error, 4)))
    results.append(('r2:', round(r2, 4)))
    results.append(('MAE:', round(mae, 4)))
    results.append(('MSE:', round(mse, 4)))
    results.append(('RMSE:', round(np.sqrt(mse), 4)))
    return results


def resultAnalysis(model, score, x_test, y_test):
    logging.info("Analysing results")

    print("Score - %.4f%s" % (score * 100, "%"))

    y_pred = model.predict(x_test)

    results = regressionResults(y_test, y_pred)
    for name, result in results:
        print(name, result)
    return results


class Model(object):

    def __init__(self, config: Config, **kwargs: Any | dict):
        self.config = config

        self.model = None

        self.dir_ = self.config.working_dir + '\\models'
        self.name = self.config.name
        self.suffix = ''
        self.extension = '.model'

        self.update(**kwargs)

    def update(self, **kwargs: Any | dict) -> None:
        """
        Updates the class attributes with given keys and values.
        :param kwargs: Any | dict[str: Any]
        :return:
            - None
        """
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            kwargs = kwargs['kwargs']

        name = self.name if 'name' not in kwargs else kwargs['name']
        logging.info(f"Updating '{name}' model attributes")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load(self) -> bool:
        """
        Loads and updates the model.
        :return:
            - completed - bool
        """
        self.model = utils.load(self.dir_, self.name + self.suffix, self.extension)
        if self.model is None:
            return False
        return True

    def save(self) -> bool:
        """
        Saves the model.
        :return:
            - completed - bool
        """
        return utils.save(self.dir_, self.name + self.suffix, self.model, self.extension)

    def gridSearch(self, param_search, x_train, y_train, n_split=10, n_jobs=-1, verbose=2):
        grid_search = gridSearch(self.model, param_search, x_train, y_train, n_split, n_jobs, verbose)
        return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

    def resultAnalysis(self, score, x_test, y_test):
        return resultAnalysis(self.model, score, x_test, y_test)
