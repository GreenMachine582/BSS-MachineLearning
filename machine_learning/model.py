from __future__ import annotations

import logging
import os
from typing import Any

from pandas import DataFrame, Series
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import utils


def load(dir_: str, name: str) -> object:
    """
    Load the model from a model file.

    :param dir_: Directory path of file, should be a str
    :param name: Name of file, should be a str
    :return: model - object
    """
    if not os.path.exists(dir_):
        logging.warning(f"Path '{dir_}' does not exist")
        return False

    name = utils.joinPath(name, ext='.model')
    model = utils.load(dir_, name)
    if model is None:
        logging.warning(f"Failed to load model '{name}'")
    return model


def save(dir_: str, name: str, model: object) -> bool:
    """
    Save the model to a model file.

    :param dir_: Directory path of file, should be a str
    :param name: Name of file, should be a str
    :param model: The classifier or estimator, should be an object
    :return: completed - bool
    """
    if not os.path.exists(dir_):
        logging.warning(f"Path '{dir_}' does not exist")
        return False

    name = utils.joinPath(name, ext='.model')
    completed = utils.save(dir_, name, model)
    if not completed:
        logging.warning(f"Failed to save model '{name}'")
    return completed


def gridSearch(model: object, param_grid: dict | list, **kwargs) -> GridSearchCV:
    """
    Exhaustive search over specified parameter values for a model.

    :param model: Model's classifier or estimator, should be an object
    :param param_grid: Enables searching over any sequence of parameter settings, should be a
     dict[str] | list[dict[str: Any]]
    :param kwargs: Additional keywords to pass to GridSearchCV
    :return: cv_results - GridSearchCV
    """
    defaults = {'n_jobs': -1, 'cv': 10, 'verbose': 2, 'return_train_score': True}
    grid_search = GridSearchCV(model, param_grid, **dict(defaults, **kwargs))
    return grid_search


class Model(object):

    def __init__(self, config: dict, **kwargs) -> None:
        """
        Create an instance of Model

        :param config: Model's configurations, should be a dict
        :param kwargs: Additional keywords to pass to update
        :return: None
        """
        self.model: Any = None
        self.dir_: str = ''
        self.name: str = ''

        self.update(**dict(config, **kwargs))

    def update(self, **kwargs) -> None:
        """
        Update the instance attributes.

        :key model: Model's classifier or estimator, should be an object
        :key dir_: Model's path directory, should be a str
        :key name: Model's name, should be a str
        :return: None
        """
        utils.update(self, kwargs)
        logging.info(f"Updated model '{self.name}' attributes")

    def load(self) -> bool:
        """
        Load the model.

        :return: completed - bool
        """
        model = load(self.dir_, self.name)
        if model is None:
            return False
        self.model = model
        return True

    def save(self) -> bool:
        """
        Save the model.

        :return: completed - bool
        """
        utils.makePath(self.dir_)
        return save(self.dir_, self.name, self.model)

    def gridSearch(self, param_grid: dict | list, X: DataFrame, y: Series) -> GridSearchCV:
        """
        Grid searches the model then fits with given data.

        :param param_grid: Enables searching over any sequence of parameter settings, should be a
         dict[str] | list[dict[str: Any]]
        :param X: Independent features, should be a DataFrame
        :param y: Dependent variables, should be a Series
        :return: results_cv - GridSearchCV
        """
        cv_results = gridSearch(self.model, param_grid, cv=TimeSeriesSplit(10))
        cv_results.fit(X, y)
        return cv_results
