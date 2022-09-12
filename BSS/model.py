from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn import metrics

from BSS import utils, Config


def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score


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


class Model(object):

    def __init__(self, config: Config, **kwargs: Any | dict):
        self.config = config

        self.model = None

        self.dir_ = self.config.working_dir + '\\models'
        self.name = self.config.name
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
        Loads the model by deserialising a model file.
        :return:
            - completed - bool
        """
        path, exist = utils.checkPath(f"{self.dir_}\\{self.name}", self.extension)
        if exist:
            logging.info(f"Loading model '{path}'")
            self.model = pickle.load(open(path, "rb"))
            return True
        else:
            logging.warning(f"Missing file '{path}'")
        return False

    def save(self) -> None:
        """
        Saves the model by serialising the model object.
        :return:
            - None
        """
        if not utils.checkPath(self.dir_):
            os.makedirs(self.dir_)
        path, _ = utils.checkPath(f"{self.dir_}\\{self.name}", self.extension)

        logging.info(f"Saving file '{path}'")
        pickle.dump(self.model, open(path, "wb"))

    def gridSearch(self, param_search, x_train, y_train, n_jobs=-1):
        logging.info("Grid Searching model hyperparameters")
        # rmse_score = metrics.make_scorer(rmse, greater_is_better=False)
        tscv = TimeSeriesSplit(n_splits=10)
        gsearch = GridSearchCV(estimator=self.model, cv=tscv, param_grid=param_search, n_jobs=n_jobs)
        gsearch.fit(x_train, y_train)
        self.model = gsearch.best_estimator_
        return gsearch.best_score_, gsearch.best_params_

    def resultAnalysis(self, score, x_test, y_test):
        logging.info("Analysing results")

        print("Score - %.4f%s" % (score * 100, "%"))

        y_pred = self.model.predict(x_test)

        results = regressionResults(y_test, y_pred)
        for name, result in results:
            print(name, result)
        return results
