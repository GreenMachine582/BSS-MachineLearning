from __future__ import annotations

import os


def checkPath(path: str, extension: str = '') -> tuple:
    """
    Adds an extension if not already included in path, then checks if path exists.
    :param path: str
    :param extension: str
    :return:
        - path, exist - tuple[str]
    """
    if os.path.splitext(path)[1] != extension:
        path = path + extension
    exist = True if os.path.exists(path) else False
    return path, exist
