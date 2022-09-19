from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any


def checkPath(*args, extension: str = '') -> tuple:
    """
    Joins the paths together, adds an extension if not already included in path,
    then checks if path exists.
    :param extension: str
    :return:
        - path_, exist - tuple[str, bool]
    """
    path_ = joinPath(*list(args), extension=extension)
    exist = True if os.path.exists(path_) else False
    return path_, exist


def joinPath(*args, extension: str = '') -> str:
    """
    Joins the paths together, adds an extension if not already included in path.
    :param extension: str
    :return:
        - path_ - str
    """
    path_ = os.path.join(*list(args))

    if os.path.splitext(path_)[1] != extension:
        path_ = path_ + extension
    return path_


def makePath(*args) -> str:
    """
    Checks if the path exists and creates the path when required.
    :return:
        - path_ - str
    """
    path_, exist = checkPath(*list(args))
    if not exist:
        os.makedirs(path_)
    return path_


def load(dir_: str, name: str, extension: str) -> Any | None:
    """
    Loads the data with appropriate method. Pickle will deserialise the contents
    of the file and json will load the contents.
    :param dir_: str
    :param name: str
    :param extension: str
    :return:
        - data - Any | None
    """
    data = None
    path_, exist = checkPath(dir_, name, extension=extension)
    if exist:
        logging.info(f"Loading model '{path_}'")
        try:
            if extension == '.json':
                with open(path_, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = pickle.load(open(path_, "rb"))
        except Exception as e:
            logging.warning(e)
    else:
        logging.warning(f"Missing file '{path_}'")
    return data


def save(dir_: str, name: str, data: Any, extension: str, indent: int = 4) -> bool:
    """
    Saves the data with appropriate method. Pickle will serialise the object,
    while json will dump the data with indenting to allow users to edit
    and easily view the encoded data.
    :param dir_: str
    :param name: str
    :param data: Any
    :param extension: str
    :param indent: int
    :return:
        - completed - bool
    """
    makePath(dir_)
    path_ = joinPath(dir_, name, extension=extension)

    logging.info(f"Saving file '{path_}'")
    try:
        if isinstance(data, object):
            pickle.dump(data, open(path_, "wb"))
        if extension == '.json':
            with open(path_, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent)
    except Exception as e:
        logging.warning(e)
        return False
    return True
