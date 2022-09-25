from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn import metrics

from . import utils


def gridSearch(model, param_search, x_train, y_train, n_split=10, n_jobs=-1, verbose=2):
    # TODO: Documentation and error handle
    logging.info("Grid Searching model hyperparameters")
    tscv = TimeSeriesSplit(n_splits=n_split)
    grid_search = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, n_jobs=n_jobs,
                               verbose=verbose)
    grid_search.fit(x_train, y_train)
    return grid_search


def regressionResults(y_test, y_pred):
    # TODO: Documentation and error handle
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


def resultAnalysis(model, score, X_test, y_test):
    # TODO: Documentation and error handle
    logging.info("Analysing results")

    print("Score - %.4f%s" % (score * 100, "%"))

    y_pred = model.predict(X_test)

    y_pred[y_pred < 0] = 0

    regressionResults(y_test, y_pred)


class Model(object):

    def __init__(self, config: dict, **kwargs):
        """
        Create an instance of Model

        :param config: model's configurations, should be a dict
        :key model: the model itself, should be an Any
        :key dir_: model's path directory, should be a str
        :key name: model's name, should be a str
        :return: None
        """
        self.model: Any = None
        self.dir_: str = ''
        self.name: str = ''

        self.update(**config)
        self.update(**kwargs)

    def update(self, **kwargs) -> None:
        """
        Updates the instance attributes, if given attributes are present
        in instance and match existing types.

        :key model: the model itself, should be an Any
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
        Loads the model.
        :return: completed - bool
        """
        name = utils.joinPath(self.name, ext='.model')
        self.model = utils.load(self.dir_, name)
        if self.model is None:
            logging.warning(f"Failed to load model '{self.name}'")
            return False
        return True

    def save(self) -> bool:
        """
        Saves the model.
        :return: completed - bool
        """
        utils.makePath(self.dir_)
        name = utils.joinPath(self.name, ext='.model')
        completed = utils.save(self.dir_, name, self.model)
        if not completed:
            logging.warning(f"Failed to save model '{self.name}'")
        return completed

    def gridSearch(self, param_search, x_train, y_train, n_split=10, n_jobs=-1, verbose=2):
        # TODO: Documentation and error handle
        grid_search = gridSearch(self.model, param_search, x_train, y_train, n_split=n_split, n_jobs=n_jobs,
                                 verbose=verbose)
        return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

    def resultAnalysis(self, score, x_test, y_test):
        # TODO: Documentation and error handle
        resultAnalysis(self.model, score, x_test, y_test)
