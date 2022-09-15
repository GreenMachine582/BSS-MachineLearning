from __future__ import annotations

import logging
from typing import Any

from BSS import utils


class Config(object):
    """
    Config stores default and important values that the Dataset and Model
    class can use. The end-user can save, load and update the attributes.
    """

    def __init__(self, working_dir: str = '', **kwargs: Any | dict):
        self.working_dir = working_dir
        self.name = ''
        self.suffix = ''

        self.config_dir = working_dir + '\\configs'

        self.config_extension = '.json'

        # dataset related
        self.seperator = ','
        self.target = 'cnt'
        self.names = None

        # training related
        self.split_ratio = 0.8
        self.random_seed = 0

        self.update(**kwargs)

        if self.name and not self.load():
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
        Loads the config file and updates the object attributes.
        :return:
            - completed - bool
        """
        data = utils.load(self.config_dir, self.name + self.suffix, self.config_extension)
        if data is None:
            return False
        self.update(**data)
        return True

    def save(self) -> bool:
        """
        Saves the config attributes as an indented dict in a json file, to allow
        end-users to edit and easily view the default configs.
        :return:
            - completed - bool
        """
        return utils.save(self.config_dir, self.name + self.suffix, self.__dict__, self.config_extension)
