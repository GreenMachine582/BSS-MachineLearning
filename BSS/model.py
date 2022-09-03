from __future__ import annotations

import logging
import os
import pickle

from BSS.utils import checkPath


class Model(object):

    def __init__(self, config: Config, model_name: str = None):
        self.config = config
        self.model_name = self.config.dataset_name if model_name is None else model_name

        self.model = None

    def load(self) -> bool:
        """
        Loads the model by deserialising a model file.
        :return:
            - completed - bool
        """
        path, exist = checkPath(f"{self.config.model_dir}\\{self.model_name}", self.config.model_extension)
        if exist:
            logging.info(f"Loading model '{path}'")
            self.model = pickle.load(open(path, "rb"))
            return True
        else:
            logging.warning(f"Missing file '{path}'")
        return False

    def save(self) -> bool:
        """
        Saves the model by serialising the model object.
        :return:
            - completed - bool
        """
        _, exist = checkPath(self.config.model_dir)
        if not exist:
            os.makedirs(self.config.model_dir)
        path, _ = checkPath(f"{self.config.model_dir}\\{self.model_name}", self.config.model_extension)

        logging.info(f"Saving file '{path}'")
        pickle.dump(self.model, open(path, "wb"))
        return True
