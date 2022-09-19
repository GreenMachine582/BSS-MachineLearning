from __future__ import annotations

import logging

import pandas as pd
from pandas import DataFrame
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from BSS import utils, Config


def bunchToDataframe(fetched_df: Bunch, target: str = 'target') -> DataFrame:
    """
    Creates a pandas DataFrame dataset from the SKLearn Bunch object.
    :param fetched_df: Bunch
    :param target: str
    :return:
        - df - DataFrame
    """
    logging.info("Converting Bunch to DataFrame")
    df = pd.DataFrame(data=fetched_df['data'], columns=fetched_df['feature_names'])
    df[target] = fetched_df['target']
    return df


def load(dir_: str, name: str, feature_names: list, target: str = 'target', suffix: str = '', extension: str = '.csv',
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
    path_, exist = utils.checkPath(dir_, name + suffix, extension=extension)
    if not exist:
        logging.warning(f"Missing file '{path_}'")
        logging.info("Fetching dataset from openml")
        try:
            fetched_dataset = fetch_openml(name, version=1)
        except Exception as e:
            logging.warning(e)
            return
        df = bunchToDataframe(fetched_dataset, target)
        save(dir_, name + suffix, df, extension, seperator)
    logging.info(f"Loading dataset '{path_}'")
    df = pd.read_csv(path_, names=feature_names, sep=seperator)
    return df


def save(dir_: str, name: str, df: DataFrame, extension: str = '.csv', seperator: str = ',') -> bool:
    """
    Saves the dataset to a csv file using pandas.
    :param dir_: str
    :param name: str
    :param df: DataFrame
    :param extension: str
    :param seperator: str
    :return:
        - completed - bool
    """
    utils.makePath(dir_)
    path_ = utils.joinPath(dir_, name, extension=extension)

    logging.info(f"Saving file '{path_}'")
    try:
        df.to_csv(path_, sep=seperator, index=False)
    except Exception as e:
        logging.warning(e)
        return False
    return True


def split(*args, split_ratio: float = 0.8) -> tuple:
    """
    Splits the datasets into two smaller datasets with given ratio.
    :param args: tuple[DataFrame]
    :param split_ratio: float
    :return:
        - split_df - tuple[dict[str: DataFrame]]
    """
    logging.info("Splitting dataset")
    split_df = []
    for df in args:
        size = round(df.shape[0] * split_ratio)
        split_df.append({'train': df[:size], 'test': df[size:]})
    return tuple(split_df)


def handleMissingData(df: DataFrame, fill: bool = True) -> DataFrame | None:
    """
    Handles missing values by forward and backward value filling, this is a common
    strategy for datasets with time series. Instances with remaining missing values
    will be dropped.
    :param df: DataFrame
    :param fill: bool
    :return:
        - df - DataFrame | None
    """
    logging.info(f"Handling missing values")
    if df is None:
        logging.warning(f"'DataFrame' object was expected, got {type(df)}")
        return

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

    def __init__(self, config: Config, **kwargs):
        self.config = config

        self.df = None

        self.dir_ = config.working_dir + '\\datasets'
        self.name = config.name
        self.suffix = config.suffix

        self.extension = '.csv'
        self.seperator = config.seperator
        self.feature_names = config.names
        self.target = config.target
        self.split_ratio = config.split_ratio

        self.update(**kwargs)

        if self.df is None:
            if not self.load():
                logging.error("Couldn't load the dataset")

    def update(self, **kwargs) -> None:
        """
        Updates the class attributes with given keys and values.
        :return:
            - None
        """
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            kwargs = kwargs['kwargs']

        name = self.name if 'name' not in kwargs else kwargs['name']
        logging.info(f"Updating '{name}' dataset attributes")
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logging.warning(f"No such attribute, '{key}'")

    def load(self) -> bool:
        """
        Loads the dataset and returns a completion status bool.
        :return:
            completed - bool
        """
        self.df = load(self.dir_, self.name, self.feature_names, self.target, self.suffix, self.extension,
                       self.seperator)
        if self.df is None:
            return False
        return True

    def save(self) -> bool:
        """
        Saves the dataset.
        :return:
            - completed - bool
        """
        return save(self.dir_, self.name + self.suffix, self.df, self.extension, self.seperator)

    def apply(self, handler: object | callable, *args, **kwargs) -> None:
        """
        Applies the given handler to the dataset with given arguments.
        :param handler: object | callable
        :return:
            - None
        """
        if callable(handler):
            df = handler(self.df, *args, **kwargs)
            if isinstance(df, DataFrame):
                self.df = df
            else:
                logging.warning(f"DataFrame object was expected, got '{type(df)}'")

    def split(self) -> tuple:
        """
        Splits the dataset into train and test datasets.
        :return:
            train, test - tuple[dict[str: DataFrame]]
        """
        x = self.df.drop(self.target, axis=1)  # denotes independent features
        y = self.df[self.target]  # denotes dependent variables
        return split(x, y, split_ratio=self.split_ratio)

    def handleMissingData(self, fill: bool = True) -> None:
        """
        Removes invalid instances in the dataset and fills missing values.
        :param fill: bool
        :return:
            - None
        """
        self.df = handleMissingData(self.df, fill)
