from __future__ import annotations

import logging
import os
import pickle

from BSS import utils


class Model(object):

    def __init__(self, config: Config, **kwargs: Any | dict):
        self.config = config

        self.model = None

        self.dir_ = self.config.working_dir + '\\models'
        self.name = self.config.name
        self.extension = '.model'

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
        logging.info(f"Updating '{name}' model attributes")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load(self) -> bool:
        """
        Loads the model by deserialising a model file.
        :return:
            - completed - bool
        """
        path, exist = utils.checkPath(f"{self.dir_}\\{self.name}", self.extension)
        if exist:
            logging.info(f"Loading model '{path}'")
            self.model = pickle.load(open(path, "rb"))
            return True
        else:
            logging.warning(f"Missing file '{path}'")
        return False

    def save(self) -> None:
        """
        Saves the model by serialising the model object.
        :return:
            - None
        """
        if not utils.checkPath(self.dir_):
            os.makedirs(self.dir_)
        path, _ = utils.checkPath(f"{self.dir_}\\{self.name}", self.extension)

        logging.info(f"Saving file '{path}'")
        pickle.dump(self.model, open(path, "wb"))
