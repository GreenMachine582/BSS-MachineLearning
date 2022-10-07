from __future__ import annotations

import logging
import os

import pandas as pd
from pandas import DataFrame, Series
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from . import utils


def bunchToDataframe(fetched_df: Bunch, target: str = 'target') -> DataFrame | None:
    """
    Create a pandas DataFrame dataset from the SKLearn Bunch object.

    :param fetched_df: Dataset fetched from openml, should be a Bunch
    :param target: Dataset's target feature, should be a str
    :return: df - DataFrame | None
    """
    df = pd.DataFrame(data=fetched_df['data'], columns=fetched_df['feature_names'])
    df[target] = fetched_df['target']
    logging.info("Converted Bunch to DataFrame")
    return df


def load(dir_: str, name: str, names: list = None, target: str = 'target', sep: str = ',') -> DataFrame | None:
    """
    Checks and loads a locally stored .csv dataset as a pandas DataFrame. If dataset
    was not located, it will attempt to fetch from OpenML, and convert the dataset to
    a DataFrame object.

    :param dir_: Directory path of file, should be a str
    :param name: Name of file, should be a str
    :param names: Dataset's feature names, should be a list[str] | None
    :param target: Dataset's target feature, should be a str
    :param sep: Dataset's seperator, should be a str
    :return: df - DataFrame | None
    """
    if os.path.splitext(name)[1] != '.csv':
        logging.warning(f"Filename '{name}' must include correct file extension of '.csv'")
        return False

    path_, exist = utils.checkPath(dir_, name)
    if not exist:
        logging.info("Fetching dataset from openml")
        try:
            fetched_dataset = fetch_openml(name.replace('.csv', ''), version=1)
        except Exception as e:
            logging.warning(e)
            logging.warning(f"Path '{path_}' does not exist")
            return
        logging.info("Fetched")
        df = bunchToDataframe(fetched_dataset, target)
        utils.makePath(dir_)
        save(dir_, name, df, sep=sep)

    df = pd.read_csv(path_, names=names, sep=sep)
    logging.info(f"Dataset '{name}' was loaded")
    return df


def save(dir_: str, name: str, df: DataFrame, sep: str = ',') -> bool:
    """
    Save the dataset to a csv file using pandas.

    :param dir_: Directory path of file, should be a str
    :param name: Name of file, should be a str
    :param df: The dataset itself, should be a DataFrame
    :param sep: Dataset's seperator, should be a str
    :return: completed - bool
    """
    if not os.path.exists(dir_):
        logging.warning(f"Path '{dir_}' does not exist")
        return False

    path_ = utils.joinPath(dir_, name)
    df.to_csv(path_, sep=sep, index=False)
    logging.info(f"Dataset '{path_}' was saved")
    return True


def split(X: DataFrame, y: Series, **kwargs) -> tuple:
    """
    Split the datasets into two smaller datasets with given ratio.

    :param X: independent features, should be a DataFrame
    :param y: dependent variables, should be a Series
    :param kwargs: Additional keywords to pass to train_test_split
    :return: X_train, X_test, y_train, y_test - tuple[DataFrame, DataFrame, Series, Series]
    """
    defaults = {'train_size': 0.8}
    X_train, X_test, y_train, y_test = train_test_split(X, y, **dict(defaults, **kwargs))

    logging.info("Train and test datasets have been split")
    return X_train, X_test, y_train, y_test


def handleMissingData(df: DataFrame, fill: bool = True) -> DataFrame:
    """
    Handle missing values by forward and backward value filling, this is a common
    strategy for datasets with time series. Instances with remaining missing values
    will be dropped.

    :param df: the dataset itself, should be a DataFrame
    :param fill: fill missing data with surrounding values, should be a bool
    :return: df - DataFrame
    """
    logging.info(f"Handling missing values")

    missing_sum = df.isnull().sum()
    if missing_sum.sum() > 0:
        if fill:
            # fills the missing value with the next or previous instance value
            df.fillna(method="ffill", limit=1, inplace=True)  # forward fill
            df.fillna(method="bfill", limit=1, inplace=True)  # backward fill

        # removes remaining instances with missing values
        df.dropna(inplace=True)
    return df


class Dataset(object):

    def __init__(self, config: dict, **kwargs) -> None:
        """
        Create an instance of Dataset

        :param config: Dataset's configurations, should be a dict
        :param kwargs: Additional keywords to pass to update
        :return: None
        """
        self.df: DataFrame | None = None
        self.dir_: str = ''
        self.name: str = ''
        self.sep: str = ''
        self.names: list[str] = []
        self.target: str = ''
        self.train_size: float = 0.

        self.update(**dict(config, **kwargs))

    def update(self, **kwargs) -> None:
        """
        Update the instance attributes.

        :key df: The dataset itself, should be a DataFrame
        :key dir_: Dataset's path directory, should be a str
        :key name: Dataset's name, should be a str
        :key sep: Dataset's seperator, should be a str
        :key names: Dataset's feature names, should be a list[str]
        :key target: Dataset's target feature, should be a str
        :key train_size: Train and test split ratio, should be a float
        :return: None
        """
        utils.update(self, kwargs)
        logging.info(f"Updated dataset '{self.name}' attributes")

    def load(self) -> bool:
        """
        Load the dataset.

        :return: completed - bool
        """
        name = utils.joinPath(self.name, ext='.csv')
        df = load(self.dir_, name, target=self.target, sep=self.sep)
        if isinstance(df, DataFrame):
            self.df = df
            return True
        logging.warning(f"Failed to load dataset '{self.name}'")
        return False

    def save(self) -> bool:
        """
        Save the dataset.

        :return: completed - bool
        """
        utils.makePath(self.dir_)
        name = utils.joinPath(self.name, ext='.csv')

        completed = save(self.dir_, name, self.df, sep=self.sep)
        if not completed:
            logging.warning(f"Failed to save dataset '{self.name}'")
        return completed

    def apply(self, func: callable, *args, **kwargs) -> DataFrame | tuple | dict:
        """
        Apply the given function to the dataset with given arguments
        and keywords.

        :param func: Function to apply to the dataset, should be a callable
        :param args: Positional arguments to pass to func
        :param kwargs: Additional keywords to pass to func
        :return: results - DataFrame | tuple | dict
        """
        if not callable(func):
            logging.warning(f"'handler' callable was expected, got '{type(func)}'")
            return False

        results = func(self.df, *args, **kwargs)
        if isinstance(results, DataFrame):
            self.df = results
        elif isinstance(results, tuple) and len(results) >= 1 and isinstance(results[0], DataFrame):
            self.df = results[0]
        elif isinstance(results, dict) and 'df' in results and isinstance(results['df'], DataFrame):
            self.df = results['df']
        else:
            logging.warning(f"'DataFrame' object was expected, got '{type(results)}'")
        return results

    def split(self, **kwargs) -> tuple:
        """
        Split the dataset into train and test datasets.

        :param kwargs: Additional keywords to pass to train_test_split
        :return: X_train, X_test, y_train, y_test - tuple[DataFrame, DataFrame, Series, Series]
        """
        X, y = self.getIndependent(), self.getDependent()
        defaults = {'train_size': self.train_size}
        return split(X, y, **dict(defaults, **kwargs))

    def getIndependent(self) -> DataFrame:
        """
        Get the independent features.

        :return: independent - DataFrame
        """
        return self.df.drop(self.target, axis=1)

    def getDependent(self) -> Series:
        """
        Get the dependent variables.

        :return: dependent - Series
        """
        return self.df[self.target]
