from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from typing import Any


def checkPath(path_: str, *paths, ext: str = '') -> tuple:
    """
    Join the paths together, adds an extension if not already included
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
    Join the paths together, adds an extension if not already included
    in path.

    :param path_: Main file path, should be a str
    :param paths: Remaining file paths, should be a tuple[str]
    :param ext: file extension, should be a str
    :return: path_ - str
    """
    path_ = os.path.join(path_, *paths)
    if os.path.splitext(path_)[1] != ext:
        path_ = path_ + ext
    return path_


def makePath(path_: str, *paths) -> str:
    """
    Check if the path exists and creates the path when required.

    :param path_: Main file path, should be a str
    :param paths: Remaining file paths, should be a tuple[str]
    :return: path_ - str
    """
    path_, exist = checkPath(path_, *paths)
    if not exist:
        os.makedirs(path_)
    return path_


def update(obj: object, kwargs: dict) -> object:
    """
    Update the objects attributes, if given attributes are present
    in object and match existing data types.

    :param obj: The object that is being updated, should be an object
    :param kwargs: Keywords and values to be updated, should be a dict
    :return: obj - object
    """
    for key, value in kwargs.items():
        if not hasattr(obj, key):
            raise AttributeError(f"'{obj.__class__.__name__}' object has no attribute '{key}'")
        else:
            attr_ = getattr(obj, key)
            if isinstance(attr_, (type(value), type(None))) or value is None:
                setattr(obj, key, value)
            else:
                raise TypeError(f"'{key}': Expected type '{type(attr_).__name__}', got '{type(value).__name__}'")
    return obj


def load(dir_: str, name: str, errors: str = 'raise') -> Any:
    """
    Load the data with appropriate method. Pickle will deserialise the
    contents of the file and json will load the contents.

    :param dir_: Directory of file, should be a str
    :param name: Name of file, should be a str
    :param errors: If 'ignore', suppress errors, should be str
    :return: data - Any
    """
    if errors not in ["ignore", "raise"]:
        raise ValueError("The parameter errors must be either 'ignore' or 'raise'")

    path_, exist = checkPath(dir_, name)
    ext = os.path.splitext(name)[1]

    if not exist:
        logging.warning(f"Path '{path_}' does not exist")
        if errors == 'raise':
            warnings.warn(f"Path '{path_}' does not exist")
        return
    if not ext:
        logging.warning(f"Name '{name}' must include file extension")
        if errors == 'raise':
            warnings.warn(f"Name '{name}' must include file extension")
        return

    if ext == '.json':
        with open(path_, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        with open(path_, 'rb') as file:
            data = pickle.load(file)
    logging.info(f"File '{name}' data was loaded")
    return data


def save(dir_: str, name: str, data: Any, indent: int = 4, errors: str = 'raise') -> bool:
    """
    Save the data with appropriate method. Pickle will serialise the
    object, while json will dump the data with indenting to allow users
    to edit and easily view the encoded data.

    :param dir_: Directory of file, should be a str
    :param name: Name of file, should be a str
    :param data: Data to be saved, should be an Any
    :param indent: Data's indentation within the file, should be an int
    :param errors: If 'ignore', suppress errors, should be str
    :return: completed - bool
    """
    if errors not in ["ignore", "raise"]:
        raise ValueError("The parameter errors must be either 'ignore' or 'raise'")

    path_ = joinPath(dir_, name)
    ext = os.path.splitext(name)[1]

    if not os.path.exists(dir_):
        logging.warning(f"Path '{dir_}' does not exist")
        if errors == 'raise':
            warnings.warn(f"Path '{dir_}' does not exist")
        return False
    if not ext:
        logging.warning(f"File '{name}' must include file extension in name")
        if errors == 'raise':
            warnings.warn(f"File '{name}' must include file extension in name")
        return False

    if ext == '.json':
        with open(path_, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent)
    elif isinstance(data, object):
        with open(path_, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    else:
        logging.warning(f"Saving method was not determined, failed to save file")
        if errors == 'raise':
            warnings.warn(f"Saving method was not determined, failed to save file")
        return False
    logging.info(f"File '{name}' was saved")
    return True


def _plotBox(ax, results: dict, target: str, title: str = ''):
    """
    Plots a boxplot and title to the given figure or axes.

    :param ax: Can be the figure or and axes
    :param results: Results from compareModels, should be a dict[str: dict[str: ndarray]]
    :param target: The target feature, should be a str
    :param title: The title of the plot, should be a str
    :return: ax
    """
    scores = [results[name][target] for name in results]
    ax.boxplot(scores, labels=[name for name in results])
    ax.suptitle(title)
    return ax


def _plotBar(ax, x: list, y: list, title: str = ''):
    """
    Plots a bar graph and title to the given figure or axes.

    :param ax: Can be the figure or and axes
    :param x: X-axis labels, should be a list[str]
    :param y: Y-axis values, should be a list[int | float]
    :param title: The title of the plot, should be a str
    :return: ax
    """
    ax.bar(x, y)
    diff = (max(y) - min(y))
    if diff != 0:
        ax.set_ylim(min(y) - (diff * 0.1), max(y) + (diff * 0.1))
    ax.set(title=title)
    return ax
