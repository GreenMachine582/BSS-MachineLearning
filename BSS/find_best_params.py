from __future__ import annotations

import logging
import os

from sklearn import ensemble, tree
from sklearn.neural_network import MLPRegressor

import BSS
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def getGradientBoostingRegressor(random_state):
    # 85.16% - {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1000, 'subsample': 0.5}
    param_search = {'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 10, 20, 25, 50],
                    'n_estimators': [500, 800, 1000, 1200],
                    'subsample': [0.10, 0.5, 0.7, 1.0]}
    return ensemble.GradientBoostingRegressor(random_state=random_state), param_search


def getRandomForestRegressor(random_state):
    param_search = [{'criterion': ['squared_error', 'absolute_error', 'poisson']},
                    {'max_depth': [3, 10, 20, 25, 50],
                     'max_features': ['sqrt', 'log2', 'auto', 1.0]}]
    return ensemble.RandomForestRegressor(random_state=random_state), param_search


def getRandomForestClassifier(random_state):
    param_search = [{'criterion': ['gini', 'entropy', 'log_loss']},
                    {'max_depth': [3, 10, 20, 25, 50],
                     'max_features': ['sqrt', 'log2', 'auto', None]}]
    return ensemble.RandomForestRegressor(random_state=random_state), param_search


def getMLPRegressor(random_state):
    param_search = {"hidden_layer_sizes": [(1,), (50,)],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "solver": ["lbfgs", "sgd", "adam"],
                    "alpha": [0.00005, 0.0005]}
    return MLPRegressor(max_iter=1200, random_state=random_state), param_search


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

    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    logging.info('Selecting model')

    while True:
        print("""
        0 - Back
        1 - Gradient Boosting Regressor
        2 - MLP Regressor
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
            else:
                print("\nPlease enter a valid choice!")

    model = ml.Model(config.model, model=model)

    cv_results = model.gridSearch(param_search, X_train, y_train)
    print('The best estimator:', cv_results.best_estimator_)
    print('The best score: %.2f%s' % (cv_results.best_score_ * 100, '%'))
    print('The best params:', cv_results.best_params_)

    model.update(model=ensemble.RandomForestRegressor(**cv_results.best_params_))
    model.fit(X_train, y_train)
    model.save()

    y_pred = model.predict(X_test)
    ml.resultAnalysis(y_test, y_pred)

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
