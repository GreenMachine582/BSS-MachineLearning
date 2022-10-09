from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pandas import DataFrame, Series
from sklearn import ensemble, linear_model, tree, neighbors, neural_network
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

import BSS
import machine_learning as ml
from machine_learning import Config, Dataset


def searchCV(model: dict, X_train: DataFrame, y_train: Series,
             random_state: int | None = None) -> GridSearchCV | RandomizedSearchCV:
    # TODO: Fix documentation
    cv_results = RandomizedSearchCV(model['base'], model['grid_params'], n_iter=100, n_jobs=-1,
                                    random_state=random_state, cv=TimeSeriesSplit(10), verbose=2)
    # cv_results = GridSearchCV(model['base'], model['grid_params'], n_jobs=-1, cv=TimeSeriesSplit(10), verbose=2)
    cv_results.fit(X_train, y_train)
    print('\n\t', model['fullname'])
    print('The best estimator:', cv_results.best_estimator_)
    print('The best score:', cv_results.best_score_)
    print('The best params:', cv_results.best_params_)
    return cv_results


def getGradientBoostingRegressor() -> Any:
    # TODO: Fix documentation
    # Score: 0.991709212819522
    best_params = {'criterion': 'squared_error',
                   'learning_rate': 0.078,
                   'max_depth': 2,
                   'n_estimators': 1050,
                   'subsample': 0.3}
    grid_params = {'criterion': ['friedman_mse', 'squared_error'],
                   'learning_rate': [0.002 * (i + 1) for i in range(100)],
                   'max_depth': range(2, 51, 2),
                   'n_estimators': range(50, 1200, 50),
                   'subsample': [0.1 * (i + 1) for i in range(10)]}

    estimator = {'name': 'GBR',
                 'fullname': "Gradient Boosting Regressor",
                 'type': 'estimator',
                 'base': ensemble.GradientBoostingRegressor(),
                 'best_params': best_params,
                 'grid_params': grid_params}
    return estimator


def getRandomForestRegressor():
    # TODO: Fix documentation
    # Score: 0.9618887559940144
    best_params = {'criterion': 'absolute_error',
                   'max_depth': 50,
                   'max_features': 0.5}
    grid_params = {'criterion': ['squared_error', 'absolute_error', 'poisson'],
                   'max_depth': range(2, 51, 2),
                   'max_features': ['sqrt', 'log2', 2, 1, 0.5]}

    estimator = {'name': 'RFR',
                 'fullname': "Random Forest Regressor",
                 'type': 'estimator',
                 'base': ensemble.RandomForestRegressor(),
                 'best_params': best_params,
                 'grid_params': grid_params}
    return estimator


# def getKNeighborsRegressor():
#     # TODO: Fix documentation
#     # Score: 0.9727944845559323
#     best_params = {'algorithm': 'auto',
#                    'n_neighbors': 1,
#                    'p': 5,
#                    'weights': 'uniform'}
#     grid_params = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#                    'n_neighbors': range(1, 15, 1),
#                    'p': [1, 2, 5],
#                    'weights': ['uniform', 'distance']}
#
#     estimator = {'name': 'KNR',
#                  'fullname': "K-Neighbors Regressor",
#                  'type': 'estimator',
#                  'base': neighbors.KNeighborsRegressor(),
#                  'best_params': best_params,
#                  'grid_params': grid_params}
#     return estimator


def getMLPRegressor():
    # TODO: Fix documentation
    # Score: 0.9999996740724255
    best_params = {'activation': 'identity',
                   'learning_rate': 'adaptive',
                   'max_iter': 2200,
                   'solver': 'adam'}
    grid_params = {'activation': ['identity', 'logical', 'tanh', 'relu'],
                   'learning_rate': ['constant', 'invscaling', 'adaptive'],
                   'max_iter': range(1000, 3001, 50),
                   'solver': ['lbfgs', 'sgd', 'adam']}

    estimator = {'name': 'KNR',
                 'fullname': "K-Neighbors Regressor",
                 'type': 'estimator',
                 'base': neural_network.MLPRegressor(),
                 'best_params': best_params,
                 'grid_params': grid_params}
    return estimator


def getDecisionTreeRegressor():
    # TODO: Fix documentation
    # Score: 0.9243407687412368
    best_params = {'criterion': 'absolute_error',
                   'max_depth': 10,
                   'max_features': None,
                   'max_leaf_nodes': 60,
                   'min_samples_leaf': 15,
                   'min_samples_split': 20,
                   'splitter': 'best'}
    grid_params = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                   'max_depth': range(2, 51, 2),
                   'max_features': ['sqrt', 'log2', None, 2, 1, 0.5],
                   'max_leaf_nodes': range(5, 100, 5),
                   'min_samples_leaf': range(5, 100, 5),
                   'min_samples_split': range(5, 100, 5),
                   'splitter': ['best', 'random']}

    estimator = {'name': 'DTR',
                 'fullname': "Decision Tree Regressor",
                 'type': 'estimator',
                 'base': tree.DecisionTreeRegressor(),
                 'best_params': best_params,
                 'grid_params': grid_params}
    return estimator


def findEstimatorParams(dataset: Dataset, config: Config) -> None:
    """

    :param dataset: The loaded and processed dataset, should be a Dataset
    :param config:
    :return: None
    """
    # TODO: documentation
    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    while True:
        print(f"""
        0 - Back
        1 - Gradient Boosting Regressor
        2 - Random Forest Regressor
        3 - MLP Regressor
        4 - Decision Tree Regressor
        """)
        choice = input("Which estimator model: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        if choice is not None:
            estimator = {}
            if choice == 0:
                return
            elif choice == 1:
                estimator = getGradientBoostingRegressor()
            elif choice == 2:
                estimator = getRandomForestRegressor()
            elif choice == 3:
                estimator = getMLPRegressor()
            elif choice == 4:
                estimator = getDecisionTreeRegressor()
            else:
                print("\nPlease enter a valid choice!")

            cv_results = searchCV(estimator, X_train, y_train)

            models = [('Default', estimator['base']),
                      ('Grid Searched', cv_results.best_estimator_),
                      ('Recorded Best', estimator['base'].set_params(**estimator['best_params']))]

            logging.info("Fitting and predicting")
            predictions = []
            for name, model in models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                predictions.append((name, np.clip(y_pred, 0, None)))

            BSS.compare_models.plotEstimatorResultAnalysis(y_test, predictions)
            BSS.compare_models.plotPredictions(y_train, y_test, predictions)
