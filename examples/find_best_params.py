from __future__ import annotations

import logging
import os

from sklearn.ensemble import GradientBoostingRegressor

import BSS
from examples import test

# Constants
local_dir = os.path.dirname(__file__)


def main(dir_=local_dir):
    config = BSS.Config(dir_, dataset_name='Bike-Sharing-Dataset-day', model_technique='test',
                        model_algorithm='all')

    dataset = BSS.Dataset(config)
    if not dataset.load():
        logging.error("Couldn't load a dataset")

    dataset = test.processData(config, dataset)
    dataset, x, y = test.extractFeatures(config, dataset)
    x_train, x_test, y_train, y_test = test.splitDataset(config, x, y)

    # edit these values
    param_search = {'learning_rate': [((i + 1) * 0.01) for i in range(100)],
                    'max_depth': [(i + 1) for i in range(12)],
                    'n_estimators': [((i + 1) * 100) for i in range(20)],
                    'subsample': [((i + 1) * 0.02) for i in range(50)]}

    model = BSS.Model(config, model=GradientBoostingRegressor())

    score, best_params = model.gridSearch(param_search, x_train, y_train)
    print(best_params)

    results = model.resultAnalysis(score, x_test, y_test)
    print(results)


if __name__ == '__main__':
    main()
