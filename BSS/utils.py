from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any


def joinExtension(path_: str, extension: str) -> str:
    """
    Adds an extension if not already included in path.
    :param path_: str
    :param extension: str
    :return:
        - path - str
    """
    if os.path.splitext(path_)[1] != extension:
        path_ = path_ + extension
    return path_


def checkPath(path_: str, extension: str = '') -> bool | tuple:
    """
    Adds an extension if not already included in path, then checks if path exists.
    :param path_: str
    :param extension: str
    :return:
        - path_, exist - tuple[str, bool]
        - exist - bool
    """
    if extension:
        path_ = joinExtension(path_, extension)

    exist = True if os.path.exists(path_) else False

    if extension:
        return path_, exist
    return exist


def load(dir_, name, extension) -> dict | object | None:
    """
    Loads the data with appropriate method. Pickle will deserialise the contents
    of the file and json will load the contents.
    :return:
        - data - dict | object | None
    """
    data = None
    path_, exist = checkPath(f"{dir_}\\{name}", extension)
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


def save(dir_: str, name: str, data: Any | object, extension: str, indent: int = 4) -> bool:
    """
    Saves the data with appropriate method. Pickle will serialise the object,
    while json will dump the data with indenting to allow end-users to edit
    and easily view the encoded data.
    :param dir_: str
    :param name: str
    :param data: Any | object
    :param extension: str
    :param indent: int
    """
    if not checkPath(dir_):
        os.makedirs(dir_)
    path_, _ = checkPath(f"{dir_}\\{name}", extension)

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
