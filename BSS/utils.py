from __future__ import annotations

import os


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
