from __future__ import annotations

import os


def joinExtension(path: str, extension: str) -> str:
    """
    Adds an extension if not already included in path.
    :param path: str
    :param extension: str
    :return:
        - path - str
    """
    if os.path.splitext(path)[1] != extension:
        path = path + extension
    return path


def checkPath(path: str, extension: str = '') -> bool | tuple:
    """
    Adds an extension if not already included in path, then checks if path exists.
    :param path: str
    :param extension: str
    :return:
        - path, exist - tuple[str, bool]
        - exist - bool
    """
    if extension:
        path = joinExtension(path, extension)

    exist = True if os.path.exists(path) else False

    if extension:
        return path, exist
    return exist
