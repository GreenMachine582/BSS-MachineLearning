from __future__ import annotations

import logging
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import BSS
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def getGradientBoostingRegressor(random_state):
    param_search = {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']}
    # param_search = {'learning_rate': [((i + 1) * 0.01) for i in range(10)],
    #                 'max_depth': [((i + 1) * 2) for i in range(6)],
    #                 'n_estimators': [((i + 1) * 200) for i in range(10)],
    #                 'subsample': [((i + 1) * 0.04) for i in range(25)]}
    # param_search = {'learning_rate': [0.01, 0.05, 0.75, 0.1, 0.15, 0.2],
    #                 'n_estimators': [100, 500, 800, 1000, 1200],
    #                 'subsample': [0.10, 0.5, 1.0],
    #                 'max_depth': [1, 3, 10, 50]}
    return GradientBoostingRegressor(random_state=random_state), param_search


def getMLPRegressor(random_state):
    param_search = {"hidden_layer_sizes": [(1,), (50,)],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "solver": ["lbfgs", "sgd", "adam"],
                    "alpha": [0.00005, 0.0005]}
    return MLPRegressor(max_iter=1200, random_state=random_state), param_search


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
    config = ml.Config(dir_, 'Bike-Sharing-Dataset-hour')

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        return

    dataset = BSS.processDataset(dataset)

    X, y = dataset.getIndependent(), dataset.getDependent()
    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    logging.info('Selecting model')

    while True:
        print("""
        0 - Back
        1 - Gradient Boosting Regressor
        2 - MLP Regressor
        3 - SVR
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
                model, param_search = getGradientBoostingRegressor(config.random_state)
                break
            elif choice == 2:
                model, param_search = getMLPRegressor(config.random_state)
                break
            elif choice == 3:
                model, param_search = getSVR()
                break
            else:
                print("\nPlease enter a valid choice!")

    model = ml.Model(config.model, model=model)

    cv_results = model.gridSearch(param_search, X, y)
    print('The best estimator:', cv_results.best_estimator_)
    print('The best score: %.2f%s' % (cv_results.best_score_ * 100, '%'))
    print('The best params:', cv_results.best_params_)

    model.save()

    y_pred = model.predict(X_test)
    ml.resultAnalysis(y_test, y_pred)

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
