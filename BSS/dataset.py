from __future__ import annotations

import logging
import os

import pandas as pd
from pandas import DataFrame
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from . import utils


def bunchToDataframe(fetched_df: Bunch, target: str = 'target') -> DataFrame | None:
    """
    Creates a pandas DataFrame dataset from the SKLearn Bunch object.

    :param fetched_df: dataset fetched from openml, should be a Bunch
    :param target: dataset's target feature, should be a str
    :return: df - DataFrame | None
    """
    df = pd.DataFrame(data=fetched_df['data'], columns=fetched_df['feature_names'])
    df[target] = fetched_df['target']
    logging.info("Converted Bunch to DataFrame")
    return df


def load(dir_: str, name: str, feature_names: list = None, target: str = 'target', sep: str = ',') -> DataFrame | None:
    """
    Checks and loads a locally stored .csv dataset as a pandas DataFrame. If dataset
    was not located, it will attempt to fetch from OpenML, and convert the dataset to
    a DataFrame object.

    :param dir_: directory path of file, should be a str
    :param name: name of file, should be a str
    :param feature_names: dataset's feature names, should be a list[str] | None
    :param target: dataset's target feature, should be a str
    :param sep: dataset's seperator, should be a str
    :return:
        - df - DataFrame | None
    """
    if os.path.splitext(name)[1] != '.csv':
        logging.warning(f"Filename '{name}' must include correct file extension of '.csv'")
        return False

    path_, exist = utils.checkPath(dir_, name)
    if not exist:
        try:
            fetched_dataset = fetch_openml(name.replace('.csv', ''), version=1)
        except Exception as e:
            logging.warning(f"Path '{path_}' does not exist")
            logging.warning(e)
            return
        logging.info("Fetched dataset from openml")
        df = bunchToDataframe(fetched_dataset, target)
        utils.makePath(dir_)
        save(dir_, name, df, sep=sep)

    df = pd.read_csv(path_, names=feature_names, sep=sep)
    logging.info(f"Dataset '{name}' was loaded")
    return df


def save(dir_: str, name: str, df: DataFrame, sep: str = ',') -> bool:
    """
    Saves the dataset to a csv file using pandas.

    :param dir_: directory path of file, should be a str
    :param name: name of file, should be a str
    :param df: the dataset itself, should be a DataFrame
    :param sep: dataset's seperator, should be a str
    :return: completed - bool
    """
    if not os.path.exists(dir_):
        logging.warning(f"Path '{dir_}' does not exist")
        return False

    path_ = utils.joinPath(dir_, name)
    df.to_csv(path_, sep=sep, index=False)
    logging.info(f"Dataset '{path_}' was saved")
    return True


def split(X: DataFrame, y: DataFrame, split_ratio: float = 0.8, random_seed: int = None,
          time_series: bool = False) -> tuple | None:
    """
    Splits the datasets into two smaller datasets with given ratio.

    :param X: independent features, should be a DataFrame
    :param y: dependent variables, should be a DataFrame
    :param split_ratio: train and test split ratio, should be a float
    :param random_seed: random seed, should be an int
    :param time_series: whether the dataset is a time series, should be a bool
    :return: X_train, X_test, y_train, y_test - tuple(DataFrame) | None
    """
    if X.shape[0] != y.shape[0]:
        logging.warning(f"X and y are not the same size, got: X {X.shape}, y {y.shape}")
        return None

    size = round(X.shape[0] * split_ratio)

    if time_series:
        X_train, X_test, y_train, y_test = X[:size], X[size:], y[:size], y[size:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_ratio, random_state=random_seed)

    logging.info("Train and test datasets have been split")
    return X_train, X_test, y_train, y_test


def handleMissingData(df: DataFrame, fill: bool = True) -> DataFrame:
    """
    Handles missing values by forward and backward value filling, this is a common
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

        :param config: dataset's configurations, should be a dict
        :key df: the dataset itself, should be a DataFrame
        :key dir_: dataset's path directory, should be a str
        :key name: dataset's name, should be a str
        :key sep: dataset's seperator, should be a str
        :key feature_names: dataset's feature names, should be a list[str]
        :key target: dataset's target feature, should be a str
        :key split_ratio: train and test split ratio, should be a float
        :return: None
        """
        self.df: DataFrame | None = None
        self.dir_: str = ''
        self.name: str = ''
        self.sep: str = ''
        self.feature_names: list[str] = []
        self.target: str = ''
        self.split_ratio: float = 0.

        self.update(**config)
        self.update(**kwargs)

    def update(self, **kwargs) -> None:
        """
        Updates the instance attributes, if given attributes are present
        in instance and match existing types.

        :key df: the dataset itself, should be a DataFrame
        :key dir_: dataset's path directory, should be a str
        :key name: dataset's name, should be a str
        :key sep: dataset's seperator, should be a str
        :key feature_names: dataset's feature names, should be a list[str]
        :key target: dataset's target feature, should be a str
        :key split_ratio: train and test split ratio, should be a float
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
        logging.info(f"Updated dataset '{self.name}' attributes")

    def load(self) -> bool:
        """
        Loads the dataset.

        :return: completed - bool
        """
        name = utils.joinPath(self.name, ext='.csv')
        self.df = load(self.dir_, name, target=self.target, sep=self.sep)
        if self.df is None:
            logging.warning(f"Failed to load dataset '{self.name}'")
            return False
        return True

    def save(self) -> bool:
        """
        Saves the dataset.

        :return: completed - bool
        """
        utils.makePath(self.dir_)
        name = utils.joinPath(self.name, ext='.csv')
        completed = save(self.dir_, name, self.df)
        if not completed:
            logging.warning(f"Failed to save dataset '{self.name}'")
        return completed

    def apply(self, func: object | callable, *args, **kwargs) -> bool:
        """
        Applies the given function to the dataset with given arguments
        and keywords.

        :param func: object | callable
        :return: completed - bool
        """
        if not callable(func):
            logging.warning(f"'handler' callable was expected, got '{type(func)}'")
            return False

        df = func(self.df, *args, **kwargs)
        if isinstance(df, DataFrame):
            self.df = df
            return True
        logging.warning(f"'DataFrame' object was expected, got '{type(df)}'")
        return False

    def split(self, random_seed: int) -> tuple:
        """
        Splits the dataset into train and test datasets.

        :param random_seed: random seed, should be an int
        :return: X_train, X_test, y_train, y_test - tuple[DataFrame]
        """
        return split(self.getIndependent(), self.getDependent(), split_ratio=self.split_ratio, random_seed=random_seed,
                     time_series=True)

    def getIndependent(self) -> DataFrame:
        """
        Gets the independent features.

        :return: independent - DataFrame
        """
        return self.df.drop(self.target, axis=1)

    def getDependent(self) -> DataFrame:
        """
        Gets the dependent variables.

        :return: dependent - DataFrame
        """
        return self.df[self.target]
