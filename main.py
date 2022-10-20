from __future__ import annotations

import logging
import os
import sys
from copy import deepcopy
from time import time

import BSS
import machine_learning as ml

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

    :return: None
    """
    logging.info("Exiting program - %s seconds -" % round(time() - START_TIME, 2))
    sys.exit(0)


def getProject(name: str) -> tuple:
    """

    :param name:
    :return: config, dataset - tuple[Config, Dataset]
    """
    # TODO: Documentation
    config = ml.Config(ROOT_DIR, name)

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        raise Exception("Failed to load dataset")
    dataset = BSS.processDataset(dataset, overwrite=False)
    return config, dataset


def main() -> None:
    """
    Gives the user a choice between tasks or datasets.

    :return: None
    """
    logging.info('Starting program')
    config, dataset = getProject('DC_day')

    while True:
        print("""
        0 - Quit
        1 - Process Datasets (Includes EDA)
        2 - Compare Default Estimators (CV)
        3 - Compare Default Classifiers (CV)
        4 - Compare Params (SearchCV)
        5 - Compare Best Models
        """)
        choice = input("Which option number: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        if choice is not None:
            if choice == 0:
                break
            elif choice == 1:
                BSS.process.main(ROOT_DIR)
            elif choice == 2:
                BSS.compare_models.compareEstimators(deepcopy(dataset), config)
            elif choice == 3:
                BSS.compare_models.compareClassifiers(deepcopy(dataset), config)
            elif choice == 4:
                BSS.compare_params.compareParams(deepcopy(dataset), config)
            elif choice == 5:
                BSS.compare_best.compareBest(deepcopy(dataset), config)
            else:
                print("\nPlease enter a valid choice!")
    quit_program()


if __name__ == '__main__':
    main()
