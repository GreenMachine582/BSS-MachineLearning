from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from BSS import utils, Config


def bunchToDataframe(fetched_df: Bunch, target: str) -> DataFrame:
    """
    Creates a pandas DataFrame dataset from the SKLearn Bunch object.
    :param fetched_df: Bunch
    :param target: str
    :return:
        - df - DataFrame
    """
    logging.info("Converting Bunch to DataFrame")
    df = pd.DataFrame(data=fetched_df["data"], columns=fetched_df["feature_names"])
    df[target] = fetched_df["target"]
    return df


def load(dir_: str, name: str, feature_names, target: str = 'target', suffix: str = '', extension: str = '.csv',
         seperator: str = ',') -> DataFrame | None:
    """
    Checks and loads a locally stored .csv dataset as a pandas DataFrame. If dataset
    was not located, it will attempt to fetch from OpenML, and convert the dataset to
    a DataFrame object.
    :param dir_: str
    :param name: str
    :param feature_names: list[str]
    :param target: str
    :param suffix: str
    :param extension: str
    :param seperator: str
    :return:
        - df - DataFrame | None
    """
    path, exist = utils.checkPath(f"{dir_}\\{name}{suffix}", extension)
    if not exist:
        logging.warning(f"Missing file '{path}'")
        logging.info("Fetching dataset from openml")
        try:
            fetched_dataset = fetch_openml(name, version=1)
        except Exception as e:
            logging.warning(e)
            return None
        df = bunchToDataframe(fetched_dataset, target)
        save(dir_, name, df, suffix, extension, seperator)
    logging.info(f"Loading dataset '{path}'")
    df = pd.read_csv(path, names=feature_names, sep=seperator)
    return df


def save(dir_: str, name: str, df: DataFrame, suffix: str = '', extension: str = '.csv', seperator: str = ',') -> None:
    """
    Saves the dataset to a csv file using pandas.
    :param dir_: str
    :param name: str
    :param df: DataFrame
    :param suffix: str
    :param extension: str
    :param seperator: str
    :return:
        - None
    """
    if not utils.checkPath(dir_):
        os.makedirs(dir_)
    path = utils.joinExtension(f"{dir_}\\{name}{suffix}", extension)

    logging.info(f"Saving file '{path}'")
    df.to_csv(path, sep=seperator, index=False)


def split(df: DataFrame, split_ratio: float) -> tuple:
    """
    Splits the dataset into two smaller datasets with given ratio.
    :param df: DataFrame
    :param split_ratio: float
    :return:
        - train, test - tuple[DataFrame]
    """
    logging.info("Splitting dataset")
    size = round(df.shape[0] * split_ratio)
    train = df[:size]
    test = df[size:]
    return train, test


def handleMissingData(df: DataFrame, name: str, suffix: str = '', fill: bool = True) -> DataFrame:
    """
    Handles missing values by forward and backward value filling, this is a common
    strategy for datasets with time series. Instances with remaining missing values
    will be dropped.
    :param df: DataFrame
    :param name: str
    :param suffix: str
    :param fill: bool
    :return:
        - df - DataFrame
    """
    logging.info(f"Handling missing values for '{name}{suffix}'")
    missing_sum = df.isnull().sum()
    if missing_sum.sum() > 0:
        if fill:
            # fills the missing value with the next or previous instance value
            df = df.fillna(method="ffill", limit=1)  # forward fill
            df = df.fillna(method="bfill", limit=1)  # backward fill

        # removes remaining instances with missing values
        df = df.dropna()
    return df


class Dataset(object):

    def __init__(self, config: Config, **kwargs: Any | dict):
        self.config = config

        self.df = None

        self.dir_ = self.config.working_dir + '\\datasets'
        self.name = self.config.name
        self.suffix = ''
        self.extension = '.csv'

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
        logging.info(f"Updating '{name}' dataset attributes")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load(self) -> bool:
        """
        Loads the dataset and returns a completion status bool.
        :return:
            completed - bool
        """
        self.df = load(self.dir_, self.name, self.config.names, self.config.target, self.suffix, self.extension,
                       self.config.seperator)
        if self.df is None:
            return False
        return True

    def save(self) -> None:
        """
        Saves the dataset.
        :return:
            - None
        """
        save(self.dir_, self.name, self.df, self.suffix, self.extension, self.config.seperator)

    def split(self) -> tuple:
        """
        Splits the dataset in two smaller datasets.
        :return:
            train, test - tuple[DataFrame]
        """
        return split(self.df, self.config.split_ratio)

    def handleMissingData(self, fill: bool = True) -> None:
        """
        Removes invalid instances in the dataset and fills missing values.
        :param fill: bool
        :return:
            - None
        """
        self.df = handleMissingData(self.df, self.name, self.suffix, fill)
