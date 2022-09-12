from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from BSS import utils, Config


class Dataset(object):

    def __init__(self, config: Config, **kwargs: Any | dict):
        self.config = config

        self.df = None

        self.dir_ = self.config.working_dir + '\\datasets'
        self.name = self.config.dataset_name
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

    def bunchToDataframe(self, fetched_dataset: Bunch) -> DataFrame:
        """
        Creates a pandas DataFrame dataset from the SKLearn Bunch object.
        :param fetched_dataset: Bunch
        :return:
            - df - DataFrame
        """
        logging.info("Converting Bunch to DataFrame")
        df = pd.DataFrame(data=fetched_dataset["data"], columns=fetched_dataset["feature_names"])
        df[self.config.target] = fetched_dataset["target"]
        return df

    def load(self) -> bool:
        """
        Checks and loads a locally stored .csv dataset as a pandas DataFrame. If dataset
        was not located, it will attempt to fetch from OpenML, and convert the dataset to
        a DataFrame object.
        :return:
            - completed - bool
        """
        path, exist = utils.checkPath(f"{self.dir_}\\{self.name}{self.suffix}", self.extension)
        if not exist:
            logging.warning(f"Missing file '{path}'")
            logging.info("Fetching dataset from openml")
            try:
                fetched_dataset = fetch_openml(self.name, version=1)
            except Exception as e:
                logging.warning(e)
                return False
            self.df = self.bunchToDataframe(fetched_dataset)
            self.save()
        logging.info(f"Loading dataset '{path}'")
        self.df = pd.read_csv(path, names=self.config.names, sep=self.config.seperator)
        return True

    def save(self) -> None:
        """
        Saves the dataset to a csv file using pandas.
        :return:
            - None
        """
        if not utils.checkPath(self.dir_):
            os.makedirs(self.dir_)
        path = utils.joinExtension(f"{self.dir_}\\{self.name}{self.suffix}", self.extension)

        logging.info(f"Saving file '{path}'")
        self.df.to_csv(path, sep=self.config.seperator, index=False)

    def handleMissingData(self) -> None:
        """
        Handles missing values by forward and backward value filling, this is a common
        strategy for datasets with time series. Instances with remaining missing values
        will be dropped.
        :return:
            - None
        """
        logging.info(f"Handling missing values for '{self.name}{self.suffix}'")
        if self.df is None:
            return

        missing_sum = self.df.isnull().sum()
        if missing_sum.sum() > 0:
            # fills the missing value with the next or previous instance value
            self.df = self.df.fillna(method="ffill", limit=1)  # forward fill
            self.df = self.df.fillna(method="bfill", limit=1)  # backward fill

            # removes remaining instances with missing values
            self.df = self.df.dropna()
