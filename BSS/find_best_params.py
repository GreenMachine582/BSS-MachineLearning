from __future__ import annotations

import logging
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

import BSS
from BSS.test import extractFeatures

# Constants
local_dir = os.path.dirname(__file__)


def getGradientBoostingRegressor():
    param_search = {'learning_rate': [((i + 1) * 0.01) for i in range(10)],
                    'max_depth': [((i + 1) * 2) for i in range(6)],
                    'n_estimators': [((i + 1) * 200) for i in range(10)],
                    'subsample': [((i + 1) * 0.04) for i in range(25)]}
    return GradientBoostingRegressor(), param_search


def getSVR():
    param_search = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                    'C': [1, 5, 10],
                    'degree': [3, 8],
                    'coef0': [0.01, 10, 0.5],
                    'gamma': ('auto', 'scale')},
    return SVR(), param_search


def main(dir_: str = local_dir):
    """


    :param dir_: project's path directory, should be a str
    :return: None
    """
    # TODO: Documentation and error handle
    name = 'Bike-Sharing-Dataset-day'
    config = BSS.Config(dir_, name)
    if not config.load():
        config.save()

    # Loads the BSS dataset
    dataset = BSS.Dataset(config.dataset, name=name + '-pre-processed')
    if not dataset.load():
        return

    # Process the dataset
    dataset.apply(BSS.processData)

    dataset.apply(extractFeatures, dataset.target)

    X_train, X_test, y_train, y_test = dataset.split(config.model['random_seed'])

    logging.info('Selecting model')

    while True:
        print("""
        0 - Back
        1 - Gradient Boosting Regressor
        2 - SVR
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
            elif choice == 2:
                model, param_search = getSVR()
                break
            else:
                print("\nPlease enter a valid choice!")

    model = BSS.Model(config.model, model=model)

    best_estimator, best_score, best_params = model.gridSearch(param_search, X_train, y_train)

    print('The best estimator:', best_estimator)
    print('The best score:', best_score)
    print('The best params:', best_params)
    model.update(model=best_estimator)

    model.resultAnalysis(best_score, X_test, y_test)

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
