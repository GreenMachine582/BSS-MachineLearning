from __future__ import annotations

import logging

from . import utils


class Config(object):
    """
    Config stores default and important values that the Dataset and Model
    class can use. The end-user can save, load and update the attributes.
    """

    def __init__(self, dir_: str, name: str, **kwargs) -> None:
        """
        Create an instance of Dataset.

        :param dir_: dataset's path directory, should be a str
        :param name: dataset's name, should be a str
        :param kwargs: Additional keywords to pass to update
        :return: None
        """
        # Default configuration for Config
        self.dir_ = utils.joinPath(dir_, 'configs')
        self.name = name
        self.random_state = 0

        # Default configuration for Dataset
        self.dataset = {'dir_': utils.joinPath(dir_, 'datasets'),
                        'name': name,
                        'sep': ',',
                        'names': [],
                        'target': 'target',
                        'train_size': 0.8}

        # Default configuration for Model
        self.model = {'dir_': utils.joinPath(dir_, 'models'),
                      'name': name}

        self.update(**kwargs)

        if not self.load():
            self.save()

    def update(self, **kwargs) -> None:
        """
        Update the instance attributes.

        :key dir_: dataset's path directory, should be a str
        :key name: dataset's name, should be a str
        :key dataset: config for dataset, should be a dict
        :key model: config for model, should be a dict
        :return: None
        """
        utils.update(self, kwargs)
        logging.info(f"Updated config '{self.name}' attributes")

    def load(self) -> bool:
        """
        Load the config file and updates the object attributes.

        :return: completed - bool
        """
        name = utils.joinPath(self.name, ext='.json')
        data = utils.load(self.dir_, name, errors='ignore')
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
        utils.makePath(self.dir_)
        name = utils.joinPath(self.name, ext='.json')
        completed = utils.save(self.dir_, name, self.__dict__)
        if not completed:
            logging.warning(f"Failed to save config '{self.name}'")
        return completed
