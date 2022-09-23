from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any


def checkPath(path_: str, *paths, ext: str = '') -> tuple:
    """
    Joins the paths together, adds an extension if not already included
    in path, then checks if path exists.

    :param path_: main file path, should be a str
    :param paths: remaining file paths, should be a tuple[str]
    :param ext: file extension, should be a str
    :return: path_, exist - tuple[str, bool]
    """
    path_ = joinPath(path_, *paths, ext=ext)
    exist = True if os.path.exists(path_) else False
    return path_, exist


def joinPath(path_: str, *paths, ext: str = '') -> str:
    """
    Joins the paths together, adds an extension if not already included
    in path.

    :param path_: main file path, should be a str
    :param paths: remaining file paths, should be a tuple[str]
    :param ext: file extension, should be a str
    :return: path_ - str
    """
    path_ = os.path.join(path_, *paths)
    if os.path.splitext(path_)[1] != ext:
        path_ = path_ + ext
    return path_


def makePath(path_: str, *paths) -> str:
    """
    Checks if the path exists and creates the path when required.

    :param path_: main file path, should be a str
    :param paths: remaining file paths, should be a tuple[str]
    :return: path_ - str
    """
    path_, exist = checkPath(path_, *paths)
    if not exist:
        os.makedirs(path_)
    return path_


def load(dir_: str, name: str) -> Any:
    """
    Loads the data with appropriate method. Pickle will deserialise the
    contents of the file and json will load the contents.

    :param dir_: directory of file, should be a str
    :param name: name of file, should be a str
    :return: data - Any
    """
    path_, exist = checkPath(dir_, name)
    ext = os.path.splitext(name)[1]

    if not exist:
        logging.warning(f"Path '{path_}' does not exist")
        return
    if not ext:
        logging.warning(f"Name '{name}' must include file extension")
        return

    if ext == '.json':
        with open(path_, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = pickle.load(open(path_, "rb"))
    logging.info(f"File '{name}' data was loaded")
    return data


def save(dir_: str, name: str, data: Any, indent: int = 4) -> bool:
    """
    Saves the data with appropriate method. Pickle will serialise the
    object, while json will dump the data with indenting to allow users
    to edit and easily view the encoded data.

    :param dir_: directory path of file, should be a str
    :param name: name of file, should be a str
    :param data: data to be saved, should be an Any
    :param indent: data's indentation within the file, should be an int
    :return: completed - bool
    """
    path_ = joinPath(dir_, name)
    ext = os.path.splitext(name)[1]

    if not os.path.exists(dir_):
        logging.warning(f"Path '{dir_}' does not exist")
        return False
    if not ext:
        logging.warning(f"File '{name}' must include file extension in name")
        return False

    if ext == '.json':
        with open(path_, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
    elif isinstance(data, object):
        pickle.dump(data, open(path_, "wb"))
    else:
        logging.warning(f"Saving method was not determined, failed to save file")
        return False
    logging.info(f"File '{name}' was saved")
    return True
