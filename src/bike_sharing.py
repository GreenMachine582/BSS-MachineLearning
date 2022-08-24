from __future__ import annotations

import os
import sys
from time import time


# Constants
ROOT_DIR = os.path.dirname(__file__)
START_TIME = time()


def quit_program():
    print("--- %s seconds ---" % round(time() - START_TIME, 2))
    sys.exit()


def main():
    run = True
    while run:
        break
    quit_program()


if __name__ == '__main__':
    main()
