from __future__ import annotations

import logging
import os
import sys
from time import time
import warnings

import examples


warnings.simplefilter(action='ignore', category=FutureWarning)

# Sets up the in-built logger to record key information and save it to a text file
logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w',
                    format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - '%(message)s'")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Outputs the loggings into screen output

# Constants
ROOT_DIR = os.path.dirname(__file__)
START_TIME = time()


def quit_program() -> None:
    """
    Closes python in a safe manner.
    :return:
        - None
    """
    logging.info("Exiting program - %s seconds -" % round(time() - START_TIME, 2))
    sys.exit(0)


def main() -> None:
    """
    Gives the user a choice between tasks or datasets.
    :return:
        - None
    """
    run = True
    while run:
        print("""
        0 - Quit
        1 - Testing
        2 - Clustering (TBA)
        3 - Neural Network (TBA)
        4 - Regression (TBA)
        """)
        choice = input("Which question number: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        if choice is not None:
            if choice == 0:
                return
            elif choice == 1:
                examples.test.main()
            elif choice == 2:
                pass
            elif choice == 3:
                pass
            elif choice == 4:
                pass
            else:
                print("\nPlease enter a valid choice!")


if __name__ == '__main__':
    logging.info('Starting program')
    main()
    raise quit_program()
