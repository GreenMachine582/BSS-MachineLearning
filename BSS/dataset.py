from __future__ import annotations

import logging
import os
from typing import Any

from BSS import utils, Config

import pandas as pd
from sklearn.datasets import fetch_openml


class Dataset(object):

    def __init__(self, config: Config, **kwargs: Any | dict):
        self.config = config

        self.dataset = None

        self.dir_ = self.config.working_dir + '/datasets'
        self.name = self.config.dataset_name
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
            - dataset - DataFrame
        """
        logging.info("Converting Bunch to DataFrame")
        dataset = pd.DataFrame(data=fetched_dataset["data"], columns=fetched_dataset["feature_names"])
        dataset[self.config.target] = fetched_dataset["target"]
        return dataset

    def load(self) -> bool:
        """
        Checks and loads a locally stored .csv dataset as a pandas DataFrame. If dataset
        was not located, it will attempt to fetch from OpenML, and convert the dataset to
        a DataFrame object.
        :return:
            - completed - bool
        """
        path, exist = utils.checkPath(f"{self.dir_}/{self.name}", self.extension)
        if not exist:
            logging.warning(f"Missing file '{path}'")
            logging.info("Fetching dataset from openml")
            try:
                fetched_dataset = fetch_openml(self.name, version=1)
            except Exception as e:
                logging.warning(e)
                return False
            self.dataset = self.bunchToDataframe(fetched_dataset)
            self.save()
        logging.info(f"Loading dataset '{path}'")
        self.dataset = pd.read_csv(path, names=self.config.names, sep=self.config.seperator)
        return True

    def save(self) -> None:
        """
        Saves the dataset to a csv file using pandas.
        :return:
            - None
        """
        if not utils.checkPath(self.dir_):
            os.makedirs(self.dir_)
        path = utils.joinExtension(f"{self.dir_}/{self.name}", self.extension)

        logging.info(f"Saving file '{path}'")
        self.dataset.to_csv(path, sep=self.config.seperator, index=False)

    def handleMissingData(self) -> None:
        """
        Handles missing values by forward and backward value filling, this is a common
        strategy for datasets with time series. Instances with remaining missing values
        will be dropped.
        :return:
            - None
        """
        logging.info(f"Handling missing values for '{self.name}'")
        if self.dataset is None:
            return

        # fills the missing value with the next or previous instance value
        self.dataset = self.dataset.fillna(method="ffill", limit=1)  # forward fill
        self.dataset = self.dataset.fillna(method="bfill", limit=1)  # backward fill

        # removes remaining instances with missing values
        self.dataset = self.dataset.dropna()
