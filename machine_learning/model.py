from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pandas import DataFrame
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import utils


def gridSearch(estimator: Any, param_grid: dict | list, X: DataFrame, y: DataFrame, scoring: str | list = None,
               n_jobs: int = -1, cv: int | object = 10, verbose: int = 2):
    # TODO: Documentations and error handling
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=n_jobs, cv=cv, verbose=verbose,
                               return_train_score=True)
    grid_search.fit(X, y)
    return grid_search


def resultAnalysis(y_test, y_pred):
    # TODO: Documentation and error handle
    logging.info("Analysing results")

    explained_variance = metrics.explained_variance_score(y_test, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    print('explained_variance: %.4f' % explained_variance)
    print('mean_squared_log_error: %.4f' % mean_squared_log_error)
    print('r2: %.4f' % r2)
    print('MAE: %.4f' % mae)
    print('MSE: %.4f' % mse)
    print('RMSE: %.4f' % np.sqrt(mse))


class Model(object):

    def __init__(self, config: dict, **kwargs) -> None:
        """
        Create an instance of Model

        :param config: model's configurations, should be a dict
        :key estimator: model's estimator, should be an Any
        :key dir_: model's path directory, should be a str
        :key name: model's name, should be a str
        :return: None
        """
        self.estimator: Any = None
        self.dir_: str = ''
        self.name: str = ''

        self.update(**config)
        self.update(**kwargs)

    def update(self, **kwargs) -> None:
        """
        Updates the instance attributes, if given attributes are present
        in instance and match existing types.

        :key estimator: model's estimator, should be an Any
        :key dir_: model's path directory, should be a str
        :key name: model's name, should be a str
        :return: None
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                logging.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
            else:
                attr_ = getattr(self, key)
                if isinstance(attr_, (type(value), type(None))):
                    setattr(self, key, value)
                else:
                    logging.error(f"'{key}': got '{type(value).__name__}' but expected type is "
                                  f"'{type(attr_).__name__}'")
        logging.info(f"Updated model '{self.name}' attributes")

    def load(self) -> bool:
        """
        Loads the estimator.
        :return: completed - bool
        """
        name = utils.joinPath(self.name, ext='.model')
        self.estimator = utils.load(self.dir_, name)
        if self.estimator is None:
            logging.warning(f"Failed to load model '{self.name}'")
            return False
        return True

    def save(self) -> bool:
        """
        Saves the estimator.
        :return: completed - bool
        """
        utils.makePath(self.dir_)
        name = utils.joinPath(self.name, ext='.model')
        completed = utils.save(self.dir_, name, self.estimator)
        if not completed:
            logging.warning(f"Failed to save model '{self.name}'")
        return completed

    def gridSearch(self, param_search: dict | list, X: DataFrame, y: DataFrame) -> GridSearchCV:
        """

        :param param_search:
        :param X:
        :param y:
        :return: results_cv - GridSearchCV
        """
        # TODO: Documentation and error handling
        return gridSearch(self.estimator, param_search, X, y, cv=TimeSeriesSplit(10))

    def fit(self, X_train: DataFrame, y_train: DataFrame) -> None:
        """
        Fitting the estimator with provided training data.

        :param X_train: training independent features, should be a DataFrame
        :param y_train: training dependent variables, should be a DataFrame
        :return: None
        """
        self.estimator.fit(X_train, y_train)

    def predict(self, X_test: DataFrame) -> DataFrame:
        """
        Forms predictions using the estimator and testing data.

        :param X_test: testing independent features, should be a DataFrame
        :return: y_pred - DataFrame
        """
        return self.estimator.predict(X_test)
