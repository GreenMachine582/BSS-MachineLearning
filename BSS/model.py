from __future__ import annotations

import logging
import os
import pickle

from BSS import utils


class Model(object):

    def __init__(self, config: Config, **kwargs: dict):
        self.config = config

        self.model = None

        self._dir = self.config.model_dir
        self.name = self.config.name
        self.extension = self.config.model_extension

        self.update(**kwargs)

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

    def load(self) -> bool:
        """
        Loads the model by deserialising a model file.
        :return:
            - completed - bool
        """
        path, exist = utils.checkPath(f"{self._dir}\\{self.name}", self.extension)
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
        if not utils.checkPath(self.config.model_dir):
            os.makedirs(self.config.model_dir)
        path, _ = utils.checkPath(f"{self._dir}\\{self.name}", self.extension)

        logging.info(f"Saving file '{path}'")
        pickle.dump(self.model, open(path, "wb"))
