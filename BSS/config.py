from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from BSS import utils


class Config(object):
    """
    Config stores default and important values that the Dataset and Model
    class can use. The end-user can save, load and update the attributes.
    """

    def __init__(self, working_dir: str = '', **kwargs: Any | dict):
        self.working_dir = working_dir
        self.dataset_name = ''

        self.config_dir = working_dir + '\\configs'

        self.config_extension = '.json'

        # dataset related
        self.seperator = ','
        self.target = 'cnt'
        self.names = None

        # training related
        self.split_ratio = 0.8
        self.random_seed = 0
        self.model_technique = 'regression'
        self.model_algorithm = 'logistic'

        self.update(**kwargs)

        self.name = f"{self.dataset_name}-{self.model_technique}-{self.model_algorithm}"

        if self.dataset_name and not self.load():
            self.save()

    def update(self, **kwargs: Any | dict) -> None:
        """
        Updates the class attributes with given keys and values.
        :param kwargs: Any | dict[str: Any]
        :return:
            - None
        """
        logging.info("Updating attributes")
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            kwargs = kwargs['kwargs']

        for key, value in kwargs.items():
            setattr(self, key, value)

    def load(self) -> bool:
        """
        Attempts to load the config file using json.
        :return:
            - completed - bool
        """
        path, exist = utils.checkPath(f"{self.config_dir}\\{self.name}", self.config_extension)

        if exist:
            logging.info(f"Loading config {path}")
            with open(path, 'r', encoding='utf-8') as f:
                self.update(**json.load(f))
            return True
        else:
            logging.warning(f"Missing file '{path}'")
        return False

    def save(self) -> bool:
        """
        Saves the config attributes as an indented dict in a json file, to allow
        end-users to edit and easily view the default configs.
        :return:
            - completed - bool
        """
        if not utils.checkPath(self.config_dir):
            os.makedirs(self.config_dir)
        path, _ = utils.checkPath(f"{self.config_dir}\\{self.name}", self.config_extension)

        logging.info(f"Saving file '{path}'")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)
        return True
