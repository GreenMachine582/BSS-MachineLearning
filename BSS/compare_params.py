from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from pandas import DataFrame, Series
from sklearn import ensemble, linear_model, neural_network
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
        cv_results = RandomizedSearchCV(model.base, model.grid_params, n_iter=100, n_jobs=-1, random_state=random_state,
                                        cv=TimeSeriesSplit(10), verbose=2)
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
    best_params = {'criterion': 'friedman_mse',
                   'learning_rate': 0.01,
                   'max_depth': 65,
                   'n_estimators': 600,
                   'subsample': 0.4}
    grid_params = {'criterion': ['friedman_mse', 'squared_error'],
                   'learning_rate': [0.002 * (i + 1) for i in range(100)],
                   'max_depth': range(5, 101, 5),
                   'n_estimators': range(50, 801, 50),
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
    best_params = {'criterion': 'squared_error',
                   'max_depth': 66,
                   'max_features': 0.5,
                   'min_samples_split': 3,
                   'n_estimators': 50}
    grid_params = {'criterion': ['squared_error', 'absolute_error', 'poisson'],
                   'max_depth': [2 * (i + 1) for i in range(40)],
                   'max_features': ['sqrt', 'log2', 2, 1, 0.5],
                   'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1., 2, 3, 4],
                   'n_estimators': [50 * (i + 1) for i in range(20)]}

    estimator = {'name': 'RFR',
                 'fullname': "Random Forest Regressor",
                 'type_': 'estimator',
                 'base': ensemble.RandomForestRegressor(),
                 'best_params': best_params,
                 'grid_params': grid_params}
    logging.info(f"Got '{estimator['name']}' attributes")
    return estimator


def getMLPRegressor() -> dict:
    """
    Get the Random Forest Regressor and appropriate attributes.

    :return: estimator - dict[str: Any]
    """
    best_params = {'activation': 'relu',
                   'learning_rate': 'adaptive',
                   'max_iter': 2600,
                   'solver': 'adam'}
    grid_params = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                   'learning_rate': ['constant', 'invscaling', 'adaptive'],
                   'max_iter': range(2500, 4001, 50),
                   'solver': ['sgd', 'adam']}

    estimator = {'name': 'MLPR',
                 'fullname': "MLP Regressor",
                 'type_': 'estimator',
                 'base': neural_network.MLPRegressor(),
                 'best_params': best_params,
                 'grid_params': grid_params}
    logging.info(f"Got '{estimator['name']}' attributes")
    return estimator


def getGradientBoostingClassifier() -> dict:
    """
    Get the Gradient Boosting Classifier and appropriate attributes.

    :return: classifier - dict[str: Any]
    """
    best_params = {'criterion': 'friedman_mse',
                   'learning_rate': 0.108,
                   'loss': 'log_loss',
                   'max_depth': 48,
                   'max_features': None,
                   'n_estimators': 140,
                   'subsample': 0.4}
    grid_params = {'criterion': ['squared_error', 'friedman_mse'],
                   'learning_rate': [0.002 * (i + 1) for i in range(100)],
                   'loss': ['log_loss', 'exponential'],
                   'max_depth': range(2, 51, 2),
                   'max_features': ['sqrt', 'log2', None],
                   'n_estimators': range(10, 201, 10),
                   'subsample': [0.1 * (i + 1) for i in range(10)]}

    classifier = {'name': 'GBC',
                  'fullname': "Gradient Boosting Classifier",
                  'type_': 'classifier',
                  'base': ensemble.GradientBoostingClassifier(),
                  'best_params': best_params,
                  'grid_params': grid_params}
    logging.info(f"Got '{classifier['name']}' attributes")
    return classifier


def getRandomForestClassifier() -> dict:
    """
    Get the Random Forest Classifier and appropriate attributes.

    :return: classifier - dict[str: Any]
    """
    best_params = {'criterion': 'entropy',
                   'max_depth': 70,
                   'max_features': 0.5,
                   'min_samples_split': 2,
                   'n_estimators': 450}
    grid_params = {'criterion': ['gini', 'entropy', 'log_loss'],
                   'max_depth': [2 * (i + 1) for i in range(40)],
                   'max_features': ['sqrt', 'log2', 2, 1, 0.5],
                   'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1., 2, 3, 4],
                   'n_estimators': [50 * (i + 1) for i in range(20)]}

    classifier = {'name': 'RFC',
                  'fullname': "Random Forest Classifier",
                  'type_': 'classifier',
                  'base': ensemble.RandomForestClassifier(),
                  'best_params': best_params,
                  'grid_params': grid_params}
    logging.info(f"Got '{classifier['name']}' attributes")
    return classifier


def getRidgeClassifier() -> dict:
    """
    Get the Ridge Classifier and appropriate attributes.

    :return: classifier - dict[str: Any]
    """
    best_params = {'alpha': 1.54,
                   'max_iter': 1750,
                   'solver': 'cholesky'}
    grid_params = {'alpha': [0.02 * (i + 1) for i in range(100)],
                   'max_iter': range(600, 2001, 50),
                   'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

    classifier = {'name': 'RC',
                  'fullname': "Ridge Classifier",
                  'type_': 'classifier',
                  'base': linear_model.RidgeClassifier(),
                  'best_params': best_params,
                  'grid_params': grid_params}
    logging.info(f"Got '{classifier['name']}' attributes")
    return classifier


def getMLPClassifier() -> dict:
    """
    Get the MLP Classifier and appropriate attributes.

    :return: classifier - dict[str: Any]
    """
    best_params = {'activation': 'logistic',
                   'learning_rate': 'constant',
                   'max_iter': 1300,
                   'solver': 'adam'}
    grid_params = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                   'learning_rate': ['constant', 'invscaling', 'adaptive'],
                   'max_iter': range(1000, 4001, 50),
                   'solver': ['sgd', 'adam']}

    classifier = {'name': 'MLPC',
                  'fullname': "MLP Classifier",
                  'type_': 'classifier',
                  'base': neural_network.MLPClassifier(),
                  'best_params': best_params,
                  'grid_params': grid_params}
    logging.info(f"Got '{classifier['name']}' attributes")
    return classifier


def compareEstimator(estimator, dataset, config):
    results_dir = ml.utils.makePath(config.dir_, config.results_folder, f"{estimator.type_}_{estimator.name}")

    X_train, X_test, y_train, y_test = dataset.split(shuffle=False)

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

    ml.estimator.resultAnalysis(y_test, y_preds, dataset_name=dataset.name, results_dir=results_dir)
    y_preds = [(name, Series(y_pred, index=y_test.index).resample('D').sum()) for name, y_pred in y_preds]
    ml.estimator.plotPrediction(y_train.resample('D').sum(), y_test.resample('D').sum(), y_preds, ylabel=dataset.target,
                                dataset_name=dataset.name, results_dir=results_dir)


def compareClassifier(classifier, dataset, config):
    results_dir = ml.utils.makePath(config.dir_, config.results_folder, f"{classifier.type_}_{classifier.name}")

    dataset.df.drop('diff', axis=1, inplace=True)  # same feature as binary encoded target
    dataset.apply(ml.binaryEncode, dataset.target)

    X_train, X_test, y_train, y_test = dataset.split(shuffle=False)

    cv_results = searchCV(classifier, X_train, y_train)

    models = [('Default', deepcopy(classifier.base)),
              ('Grid Searched', cv_results.best_estimator_),
              ('Recorded Best', deepcopy(classifier.base).set_params(**classifier.best_params))]

    logging.info("Fitting and predicting")
    y_preds = []
    for name, model in models:
        model.fit(X_train, y_train)
        y_preds.append((name, model.predict(X_test)))

    ml.classifier.resultAnalysis(y_test, y_preds, dataset_name=dataset.name, results_dir=results_dir)

    ml.classifier.plotPrediction(y_test, y_preds, dataset_name=dataset.name, results_dir=results_dir)


def compareParams(dataset: Dataset, config: Config) -> None:
    """
    The user can select an ML technique to be grid-searched and cross-validated.
    A default, grid and best model will be fitted and predicted. The predictions
    be used for a result analysis and will be plotted.

    :param dataset: The loaded and processed dataset, should be a Dataset
    :param config: BSS configuration, should be a Config
    :return: None
    """
    logging.info("Comparing params")
    while True:
        print(f"""
        0 - Back
        Estimators:
            1 - Gradient Boosting Regressor
            2 - Random Forest Regressor
            3 - MLP Regressor
        Classifiers:
            4 - Gradient Boosting Classifier
            5 - Random Forest Classifier
            6 - Ridge Classifier
            7 - MLP Classifier
        """)
        choice = input("Which estimator model: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        model_config = None
        if choice is not None:
            if choice == 0:
                return
            elif choice == 1:
                model_config = getGradientBoostingRegressor()
            elif choice == 2:
                model_config = getRandomForestRegressor()
            elif choice == 3:
                model_config = getMLPRegressor()
            elif choice == 4:
                model_config = getGradientBoostingClassifier()
            elif choice == 5:
                model_config = getRandomForestClassifier()
            elif choice == 6:
                model_config = getRidgeClassifier()
            elif choice == 7:
                model_config = getMLPClassifier()
            else:
                print("\nPlease enter a valid choice!")

        if model_config is not None:
            model = Model(config.model, **model_config)
            if model.type_ == 'estimator':
                compareEstimator(model, dataset, config)
            elif model.type_ == 'classifier':
                compareClassifier(model, deepcopy(dataset), config)
