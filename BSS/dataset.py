from __future__ import annotations

import logging
import os

from BSS.utils import checkPath

import pandas as pd
from sklearn.datasets import fetch_openml


class Dataset(object):

    def __init__(self, config: Config, dataset_name: str = None):
        self.config = config
        self.dataset_name = self.config.dataset_name if dataset_name is None else dataset_name

        self.dataset = None

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
        path, exist = checkPath(f"{self.config.dataset_dir}\\{self.dataset_name}", self.config.dataset_extension)
        if not exist:
            logging.warning(f"Missing file '{path}'")
            logging.info("Fetching dataset from openml")
            try:
                fetched_dataset = fetch_openml(self.dataset_name, version=1)
            except Exception as e:
                logging.warning(e)
                return False
            self.dataset = self.bunchToDataframe(fetched_dataset)
            self.saveDataset()
        logging.info(f"Loading dataset '{path}'")
        self.dataset = pd.read_csv(path, names=self.config.names, sep=self.config.seperator, low_memory=False)
        return True

    def save(self) -> bool:
        """
        Saves the dataset to a csv file using pandas.
        :return:
            - completed - bool
        """
        _, exist = checkPath(self.config.dataset_dir)
        if not exist:
            os.makedirs(self.config.dataset_dir)
        path, _ = checkPath(f"{self.config.dataset_dir}\\{self.dataset_name}", self.config.dataset_extension)

        logging.info(f"Saving file '{path}'")
        self.dataset.to_csv(path, sep=self.config.seperator, index=False)
        return True
