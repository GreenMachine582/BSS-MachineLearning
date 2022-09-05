from __future__ import annotations

import logging
import os

from BSS import utils

import pandas as pd
from sklearn.datasets import fetch_openml


class Dataset(object):

    def __init__(self, config: Config, **kwargs: dict):
        self.config = config

        self.dataset = None

        self._dir = self.config.dataset_dir
        self.name = self.config.dataset_name
        self.extension = self.config.dataset_extension

        self.update(**kwargs)

        self.load()

    def update(self, **kwargs: dict) -> None:
        """
        Updates the class attributes with given keys and values.
        :param kwargs: dict[str: Any]
        :return:
            - None
        """
        logging.info("Updating attributes")
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            kwargs = kwargs['kwargs']

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
        path, exist = utils.checkPath(f"{self._dir}\\{self.name}", self.extension)
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
        if not utils.checkPath(self._dir):
            os.makedirs(self._dir)
        path = utils.joinExtension(f"{self._dir}\\{self.name}", self.extension)

        logging.info(f"Saving file '{path}'")
        self.dataset.to_csv(path, sep=self.config.seperator, index=False)
