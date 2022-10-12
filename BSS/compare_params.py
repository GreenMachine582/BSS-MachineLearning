from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from pandas import DataFrame, Series
from sklearn import ensemble, neural_network, neighbors
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

import machine_learning as ml
from machine_learning import Config, Dataset, Model


def searchCV(model: Model, X_train: DataFrame, y_train: Series, display: bool = True, search_method: str = 'randomised',
             random_state: int | None = None) -> GridSearchCV | RandomizedSearchCV:
    """
    Search through param grids with two distinct search methods, and
    incorporates cross-validation as it searches for the highest performing
    model.

    :param model: Model to be searched and cross-validated, should be a Model
    :param X_train: Training independent features, should be a DataFrame
    :param y_train: Training dependent variables, should be a Series
    :param display: Whether to display the results, should be a bool
    :param search_method: If 'randomised', RandomizedSearchCV will be used, if 'grid',
     GridSearchCV will be used, should be a str
    :param random_state: Also known as random seed, should be an int
    :return: cv_results - GridSearchCV | RandomizedSearchCV
    """
    logging.info(f"Grid searching '{model.name}'")
    if search_method == "randomised":
        cv_results = RandomizedSearchCV(model.base, model.grid_params, n_iter=100, n_jobs=-1,
                                        random_state=random_state, cv=TimeSeriesSplit(10), verbose=2)
    elif search_method == "grid":
        cv_results = GridSearchCV(model.base, model.grid_params, n_jobs=-1, cv=TimeSeriesSplit(10), verbose=2)
    else:
        raise ValueError("The parameter 'search_method' must be either 'randomised' or 'grid'")

    cv_results.fit(X_train, y_train)

    if display:
        print('\n\t', model.name)
        print('The best estimator:', cv_results.best_estimator_)
        print('The best score:', cv_results.best_score_)
        print('The best params:', cv_results.best_params_)
    return cv_results


def getGradientBoostingRegressor() -> dict:
    """
    Get the Gradient Boosting Regressor and appropriate attributes.

    :return: estimator - dict[str: Any]
    """
    best_params = {'criterion': 'squared_error',
                   'learning_rate': 0.078,
                   'max_depth': 2,
                   'n_estimators': 600,
                   'subsample': 0.3}
    grid_params = {'criterion': ['friedman_mse', 'squared_error'],
                   'learning_rate': [0.002 * (i + 1) for i in range(100)],
                   'max_depth': range(2, 51, 2),
                   'n_estimators': range(50, 401, 50),
                   'subsample': [0.1 * (i + 1) for i in range(10)]}

    estimator = {'name': 'GBR',
                 'fullname': "Gradient Boosting Regressor",
                 'type_': 'estimator',
                 'base': ensemble.GradientBoostingRegressor(),
                 'best_params': best_params,
                 'grid_params': grid_params}
    logging.info(f"Got '{estimator['name']}' attributes")
    return estimator


def getRandomForestRegressor() -> dict:
    """
    Get the Random Forest Regressor and appropriate attributes.

    :return: estimator - dict[str: Any]
    """
    best_params = {'criterion': 'absolute_error',
                   'max_depth': 50,
                   'max_features': 0.5}
    grid_params = {'criterion': ['squared_error', 'absolute_error', 'poisson'],
                   'max_depth': range(2, 101, 2),
                   'max_features': ['sqrt', 'log2', 2, 1, 0.5]}

    estimator = {'name': 'RFR',
                 'fullname': "Random Forest Regressor",
                 'type_': 'estimator',
                 'base': ensemble.RandomForestRegressor(),
                 'best_params': best_params,
                 'grid_params': grid_params}
    logging.info(f"Got '{estimator['name']}' attributes")
    return estimator


def getKNeighborsRegressor() -> dict:
    """
    Get the Random Forest Regressor and appropriate attributes.

    :return: estimator - dict[str: Any]
    """
    best_params = {'algorithm': 'ball_tree',
                   'n_neighbors': 2,
                   'p': 2,
                   'weights': 'distance'}
    grid_params = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                   'n_neighbors': range(1, 16, 1),
                   'p': [1, 2, 5],
                   'weights': ['uniform', 'distance']}

    estimator = {'name': 'KNR',
                 'fullname': "K-Neighbors Regressor",
                 'type_': 'estimator',
                 'base': neighbors.KNeighborsRegressor(),
                 'best_params': best_params,
                 'grid_params': grid_params}
    logging.info(f"Got '{estimator['name']}' attributes")
    return estimator


def getMLPRegressor() -> dict:
    """
    Get the Random Forest Regressor and appropriate attributes.

    :return: estimator - dict[str: Any]
    """
    best_params = {'activation': 'identity',
                   'learning_rate': 'adaptive',
                   'max_iter': 2200,
                   'solver': 'adam'}
    grid_params = {'activation': ['identity', 'logical', 'tanh', 'relu'],
                   'learning_rate': ['constant', 'invscaling', 'adaptive'],
                   'max_iter': range(2000, 3001, 50),
                   'solver': ['lbfgs', 'sgd', 'adam']}

    estimator = {'name': 'MLPR',
                 'fullname': "MLP Regressor",
                 'type_': 'estimator',
                 'base': neural_network.MLPRegressor(),
                 'best_params': best_params,
                 'grid_params': grid_params}
    logging.info(f"Got '{estimator['name']}' attributes")
    return estimator


def findEstimatorParams(dataset: Dataset, config: Config) -> None:
    """
    The user can select an ML technique to be grid-searched and cross-validated.
    A default, grid and best model will be fitted and predicted. The predictions
    be used for a result analysis and will be plotted.

    :param dataset: The loaded and processed dataset, should be a Dataset
    :param config: BSS configuration, should be a Config
    :return: None
    """
    X_train, X_test, y_train, y_test = dataset.split(shuffle=False)

    while True:
        print(f"""
        0 - Back
        1 - Gradient Boosting Regressor
        2 - Random Forest Regressor
        3 - K-Neighbors Regressor
        4 - MLP Regressor
        """)
        choice = input("Which estimator model: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        estimator = None
        if choice is not None:
            if choice == 0:
                return
            elif choice == 1:
                estimator = getGradientBoostingRegressor()
            elif choice == 2:
                estimator = getRandomForestRegressor()
            elif choice == 3:
                estimator = getKNeighborsRegressor()
            elif choice == 4:
                estimator = getMLPRegressor()
            else:
                print("\nPlease enter a valid choice!")

        if estimator is not None:
            estimator = Model(config.model, **estimator)

            results_dir = ml.utils.makePath(config.dir_, config.results_folder, f"{estimator.type_}_{estimator.name}")

            cv_results = searchCV(estimator, X_train, y_train)

            models = [('Default', deepcopy(estimator.base)),
                      ('Grid Searched', cv_results.best_estimator_),
                      ('Recorded Best', deepcopy(estimator.base).set_params(**estimator.best_params))]

            logging.info("Fitting and predicting")
            y_preds = []
            for name, model in models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_preds.append((name, np.clip(y_pred, 0, None)))

            ml.estimator.resultAnalysis(y_test, y_preds, display=False, dataset_name=dataset.name, dir_=results_dir)

            y_preds = [(name, Series(y_pred, index=y_test.index).resample('D').sum()) for name, y_pred in y_preds]
            ml.estimator.plotPrediction(y_train.resample('D').sum(), y_test.resample('D').sum(), y_preds,
                                        target=dataset.target,
                                        dataset_name=dataset.name, dir_=results_dir)
