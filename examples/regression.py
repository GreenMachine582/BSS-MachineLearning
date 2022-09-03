
import logging
import os

import BSS

# Constants
local_dir = os.path.dirname(__file__)


def processData(dataset):
    print(dataset.head())


def extractFeatures():
    pass


def exploratoryDataAnalysis():
    pass


def splitDataset():
    pass


def trainModel():
    pass


def resultAnalysis():
    pass


def main():
    config = BSS.Config(local_dir, 'Bike-Sharing-Dataset-day')

    raw_dataset = BSS.Dataset(config)
    raw_dataset.load()

    if raw_dataset.dataset is None:
        logging.error("Couldn't load a dataset")

    processData(raw_dataset.dataset)


if __name__ == '__main__':
    main()
