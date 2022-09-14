from __future__ import annotations

import logging
import os

from sklearn.ensemble import GradientBoostingRegressor

import BSS
from examples import test

# Constants
local_dir = os.path.dirname(__file__)


def getGradientBoostingRegressor():
    param_search = {'learning_rate': [((i + 1) * 0.01) for i in range(10)],
                    'max_depth': [((i + 1) * 2) for i in range(6)],
                    'n_estimators': [((i + 1) * 200) for i in range(10)],
                    'subsample': [((i + 1) * 0.04) for i in range(25)]}
    return GradientBoostingRegressor(), param_search


def main(dir_=local_dir):
    config = BSS.Config(dir_, dataset_name='Bike-Sharing-Dataset-day', model_technique='test',
                        model_algorithm='all')

    dataset = BSS.Dataset(config)
    if not dataset.load():
        logging.error("Couldn't load a dataset")

    dataset = test.processData(config, dataset)
    dataset, x, y = test.extractFeatures(config, dataset)
    x_train, x_test, y_train, y_test = test.splitDataset(config, x, y)

    while True:
        print("""
        0 - Back
        1 - Gradient Boosting Regressor
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
                model, param_search = getGradientBoostingRegressor()
                break
            else:
                print("\nPlease enter a valid choice!")

    model = BSS.Model(config, model=model)

    best_estimator, best_score, best_params = model.gridSearch(param_search, x_train, y_train)

    print('The best estimator:', best_estimator)
    print('The best score:', best_score)
    print('The best params:', best_params)
    model.update(model=best_estimator)

    results = model.resultAnalysis(best_score, x_test, y_test)
    print(results)


if __name__ == '__main__':
    main()
