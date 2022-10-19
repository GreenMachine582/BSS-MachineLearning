from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from numpy import ndarray
from pandas import Series
from sklearn.inspection import permutation_importance

from . import classifier, estimator, utils


def load(dir_: str, name: str, ext: str = '.model') -> Any:
    """
    Load the model from a model file.

    :param dir_: Directory path of file, should be a str
    :param name: Name of file, should be a str
    :param ext: File extension, should be a str
    :return: model - Any
    """
    if not os.path.exists(dir_):
        logging.warning(f"Path '{dir_}' does not exist")
        return None

    name = utils.joinPath(name, ext=ext)
    model = utils.load(dir_, name)
    if model is None:
        logging.warning(f"Failed to load model '{name}'")
    return model


def save(dir_: str, name: str, model: object, ext: str = '.model') -> bool:
    """
    Save the model to a model file.

    :param dir_: Directory path of file, should be a str
    :param name: Name of file, should be a str
    :param model: The classifier or estimator, should be an object
    :param ext: File extension, should be a str
    :return: completed - bool
    """
    if not os.path.exists(dir_):
        logging.warning(f"Path '{dir_}' does not exist")
        return False

    name = utils.joinPath(name, ext=ext)
    completed = utils.save(dir_, name, model)
    if not completed:
        logging.warning(f"Failed to save model '{name}'")
    return completed


class Model(object):
    FOLDER_NAME = 'models'

    def __init__(self, config: dict, **kwargs) -> None:
        """
        Create an instance of Model

        :param config: Model's configurations, should be a dict
        :param kwargs: Additional keywords to pass to update
        :return: None
        """
        self.dir_: str = ''

        self.name: str = ''
        self.fullname: str = ''
        self.type_: str = ''
        self.base: Any = None
        self.best_params: dict = {}
        self.grid_params: dict = {}

        self.model: Any = None

        self.update(**dict(config, **kwargs))

    def update(self, **kwargs) -> None:
        """
        Update the instance attributes.

        :key dir_: Project's path directory, should be a str
        :key name: Model's name, should be a str
        :key fullname: Model's fullname, should be a str
        :key type_: If 'estimator', estimator methods will be used,
         if 'classifier', classifier methods will be used, should be str
        :key base: The model's default model, should be an Any
        :key best_params: The model's best set of params, should be a dict[str: Any]
        :key grid_params: The model's set of params to be searched, should be a
         list[dict[str: Any]] | dict[str: Any]
        :key model: Model's modified base model, can be saved, should be an Any
        :return: None
        """
        if 'type_' in kwargs and kwargs['type_'] not in ["estimator", "classifier"]:
            raise ValueError("The parameter type_ must be either 'estimator' or 'classifier'")
        utils.update(self, kwargs)
        logging.info(f"Updated model '{self.name}' attributes")

    def load(self) -> bool:
        """
        Load the model attribute.

        :return: completed - bool
        """
        model = load(utils.joinPath(self.dir_, self.FOLDER_NAME), self.name)
        if model is None:
            return False
        self.model = model
        return True

    def save(self) -> bool:
        """
        Save the model attribute.

        :return: completed - bool
        """
        if self.model is None:
            return False

        path_ = utils.makePath(self.dir_, self.FOLDER_NAME)
        return save(path_, self.name, self.model)

    def createModel(self, param_type: str = 'best', **kwargs) -> Any:
        """

        :param param_type:
        :param kwargs: Additional keywords to pass to sklearn's set_params
        :return: None
        """
        # TODO: Add documentation
        logging.info("Creating")
        if param_type == 'best':
            params = self.best_params
        elif param_type == 'grid':
            params = self.grid_params
        else:
            raise ValueError("The parameter param_type must be either 'best' or 'grid'")

        self.model = deepcopy(self.base).set_params(**dict(params, **kwargs))
        return self.model

    def plotPrediction(self, y_test: Series, y_pred: ndarray, y_train: Series = None, **kwargs) -> None:
        """
        Plot the prediction on a line graph.

        :param y_test: Testing dependent features, should be a Series
        :param y_pred: Predicted dependent variables, should be a ndarray
        :param y_train: Training independent features, should be a Series
        :key target: The predicted variables name, should be a str
        :key dataset_name: Name of dataset, should be a str
        :key dir_: Save location for figures, should be a str
        :return: None
        """
        if self.type_ == 'estimator':
            estimator.plotPrediction(y_train, y_test, (self.name, y_pred), **kwargs)
        elif self.type_ == 'classifier':
            if 'target' in kwargs:
                del kwargs['target']
            classifier.plotPrediction(y_test, (self.name, y_pred), **kwargs)

    def resultAnalysis(self, y_test: Series, y_pred: ndarray, **kwargs) -> None:
        """
        Calculate the result analysis.

        :param y_test: Testing dependent variables, should be a Series
        :param y_pred: Predicted dependent variables, should be a ndarray
        :key plot: Whether to plot the results, should be a bool
        :key display: Whether to display the results, should be a bool
        :key dataset_name: Name of dataset, should be a str
        :key dir_: Save location for figures, should be a str
        :return: results - dict[str: list[str | float]]
        """
        if self.type_ == 'estimator':
            estimator.resultAnalysis(y_test, (self.name, y_pred), **kwargs)
        elif self.type_ == 'classifier':
            classifier.resultAnalysis(y_test, (self.name, y_pred), **kwargs)

    def plotImportance(self, feature_names, X_test, y_test, dataset_name: str = '', dir_: str = '') -> None:
        """

        :param feature_names:
        :param X_test:
        :param y_test:
        :param dataset_name: Name of dataset, should be a str
        :param dir_: Save location for figures, should be a str
        :return:
        """
        # TODO: Documentation
        if self.model is None:
            logging.warning("Model attribute 'model' has not been assigned.")
            return

        if hasattr(self.model, 'feature_importances_'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            feature_importance = self.model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + 0.5
            ax1.barh(pos, feature_importance[sorted_idx], align="center")
            ax1.set_yticks(pos, np.array(feature_names)[sorted_idx])

            result = permutation_importance(self.model, X_test, y_test, n_repeats=10, n_jobs=-1)
            sorted_idx = result.importances_mean.argsort()
            ax2.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(feature_names)[sorted_idx])
            fig.suptitle(f"Feature and Permutation Importance - {dataset_name}")
            if dir_:
                plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
        elif hasattr(self.model, 'coef_'):
            fig, ax = plt.subplots(figsize=(10, 6))
            coef = Series(self.model.coef_[0], index=X_test.columns)
            imp_coef = coef.sort_values()
            rcParams['figure.figsize'] = (8.0, 10.0)
            imp_coef.plot(kind="barh")
            fig.suptitle(f"Coefficient Importance - {dataset_name}")
            if dir_:
                plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            result = permutation_importance(self.model, X_test, y_test, n_repeats=10, n_jobs=-1)
            sorted_idx = result.importances_mean.argsort()
            ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(feature_names)[sorted_idx])
            fig.suptitle(f"Permutation Importance - {dataset_name}")
            if dir_:
                plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
        plt.show()
