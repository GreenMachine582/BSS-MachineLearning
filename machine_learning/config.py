from __future__ import annotations

import logging
from typing import Any

from . import utils


class Config(object):
    """
    Config stores default and important values that the Dataset and Model
    class can use. The end-user can save, load and update the attributes.
    """
    EXT = '.json'
    FOLDER_NAME = 'configs'

    def __init__(self, dir_: str, name: str, **kwargs) -> None:
        """
        Create an instance of Config.

        :param dir_: Project directory, should be a str
        :param name: Project name, should be a str
        :param kwargs: Additional keywords to pass to update
        :return: None
        """
        # Default configuration for Config
        self.dir_: str = dir_
        self.name: str = name
        self.random_state: int = 0
        self.results_folder: str = 'results'

        # Default configuration for Dataset
        self._dataset: dict[str: Any] = {'name': name,
                                         'sep': ',',
                                         'names': [],
                                         'target': 'cnt',
                                         'train_size': 0.75}

        # Default configuration for Model
        self._model: dict[str: Any] = {'name': name}

        self.update(**kwargs)

        if not self.load():
            self.save()

    def update(self, **kwargs) -> None:
        """
        Update the instance attributes.

        :key dir_: Project directory, should be a str
        :key name: Project name, should be a str
        :key random_state: Also known as random seed, should be an int
        :key results_folder: Name of the result folder, should be a str
        :key dataset: Config for dataset, should be a dict
        :key model: Config for model, should be a dict
        :return: None
        """
        utils.update(self, kwargs)
        logging.info(f"Updated config '{self.name}' attributes")

    def load(self) -> bool:
        """
        Load the config file and updates the object attributes.

        :return: completed - bool
        """
        if not utils.checkPath(self.dir_):
            logging.warning(f"Path '{self.dir_}' does not exist")
            return False

        name = utils.joinPath(self.name, ext=self.EXT)
        data = utils.load(utils.joinPath(self.dir_, self.FOLDER_NAME), name, errors='ignore')
        if data is None:
            logging.warning(f"Failed to load config '{self.name}'")
            return False
        self.update(**data)
        return True

    def save(self) -> bool:
        """
        Save the config attributes as an indented dict in a json file, to allow
        users to edit and easily view the default configs.

        :return: completed - bool
        """
        path_ = utils.makePath(self.dir_, self.FOLDER_NAME)
        name = utils.joinPath(self.name, ext=self.EXT)
        completed = utils.save(path_, name, self.__dict__)
        if not completed:
            logging.warning(f"Failed to save config '{self.name}'")
        return completed

    @property
    def dataset(self) -> dict:
        """
        Return the dataset configuration.

        :return: dataset_config - dict[str: Any]
        """
        return dict(self._dataset, dir_=self.dir_)

    @dataset.setter
    def dataset(self, dataset_config: dict) -> None:
        """
        Updates the dataset attribute.

        :param dataset_config: Config for dataset, should be a dict
        :return: None
        """
        self._dataset = dataset_config

    @property
    def model(self) -> dict:
        """
        Return the model configuration.

        :return: model_config - dict[str: Any]
        """
        return dict(self._model, dir_=self.dir_)

    @model.setter
    def model(self, model_config: dict) -> None:
        """
        Updates the model attribute.

        :param model_config: Config for model, should be a dict
        :return: None
        """
        self._model = model_config
