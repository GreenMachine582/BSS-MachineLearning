from __future__ import annotations

from typing import Any

from pandas import DataFrame
from sklearn import ensemble, linear_model, tree
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import BSS
import machine_learning as ml
from machine_learning import Config, Dataset


def find_best_params(model: Any | Pipeline, param_grid, X_train: DataFrame, y_train: DataFrame) -> Any:
    # TODO: Fix documentation
    cv_results = GridSearchCV(model, param_grid, n_jobs=-1, cv=TimeSeriesSplit(10), verbose=2)
    cv_results.fit(X_train, y_train)
    print('The best estimator:', cv_results.best_estimator_)
    print('The best score:', cv_results.best_score_)
    print('The best params:', cv_results.best_params_)
    return cv_results.best_estimator_


def getGradientBoostingRegressor(best_params: bool = False):
    # TODO: Fix documentation
    # Score: 0.49975316083691734
    params = {}
    if best_params:
        params = {'learning_rate': 0.1,
                  'max_depth': 50,
                  'n_estimators': 1200,
                  'subsample': 0.5}
    estimator = Pipeline([('StandardScaler', StandardScaler()),
                          ('GradientBoostingRegressor', ensemble.GradientBoostingRegressor(**params))])
    return estimator


def getRandomForestRegressor(best_params: bool = False):
    # TODO: Fix documentation
    # Score: 0.47414265791835514
    params = {}
    if best_params:
        params = {'criterion': 'absolute_error',
                  'max_depth': 10,
                  'max_features': 0.5}
    estimator = Pipeline([('StandardScaler', StandardScaler()),
                          ('RandomForestRegressor', ensemble.RandomForestRegressor(**params))])
    return estimator


def getDecisionTreeRegressor(best_params: bool = False):
    # TODO: Fix documentation and param grid
    # Score:
    params = {}
    if best_params:
        params = {}
    estimator = Pipeline([('StandardScaler', StandardScaler()),
                          ('GradientBoostingRegressor', tree.DecisionTreeRegressor(**params))])

    return estimator


def getExtraTreeRegressor(best_params: bool = False):
    # TODO: Fix documentation and param grid
    # Score:
    params = {}
    if best_params:
        params = {}
    estimator = Pipeline([('StandardScaler', StandardScaler()),
                          ('ExtraTreeRegressor', tree.ExtraTreeRegressor(**params))])
    return estimator


def getGradientBoostingClassifier(best_params: bool = False):
    # TODO: Fix documentation
    # Score:
    params = {}
    if best_params:
        params = {}
    classifier = Pipeline([('StandardScaler', StandardScaler()),
                          ('GradientBoostingClassifier', ensemble.GradientBoostingClassifier(**params))])
    return classifier


def getRandomForestClassifier(best_params: bool = False):
    # TODO: Fix documentation
    # Score:
    params = {}
    if best_params:
        params = {}
    classifier = Pipeline([('StandardScaler', StandardScaler()),
                          ('RandomForestClassifier', ensemble.RandomForestClassifier(**params))])
    return classifier


def getRidgeClassifier(best_params: bool = False):
    # TODO: Fix documentation and param grid
    # Score:
    params = {}
    if best_params:
        params = {}
    classifier = Pipeline([('StandardScaler', StandardScaler()),
                          ('RidgeClassifier', linear_model.RidgeClassifier(**params))])
    return classifier


def getDecisionTreeClassifier(best_params: bool = False):
    # TODO: Fix documentation and param grid
    # Score:
    params = {}
    if best_params:
        params = {}
    classifier = Pipeline([('StandardScaler', StandardScaler()),
                          ('ExtraTreeClassifier', tree.DecisionTreeClassifier(**params))])
    return classifier


def findEstimatorParams(dataset: Dataset, config: Config) -> None:
    """

    :param dataset: The loaded and processed dataset, should be a Dataset
    :param config:
    :return: None
    """
    # TODO: documentation
    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    best = False
    while True:
        print(f"""
            0 - Back
            1 - Use best params (toggle) - {best}
            2 - GradientBoostingRegressor
            3 - RandomForestRegressor
            4 - DecisionTreeRegressor
            5 - ExtraTreeRegressor
            """)
        choice = input("Which estimator model: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        if choice is not None:
            estimator, param_grid = None, {}
            if choice == 0:
                return
            elif choice == 1:
                best = not best
            elif choice == 2:
                estimator = getGradientBoostingRegressor(best)
                param_grid = {'GradientBoostingRegressor__learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                              'GradientBoostingRegressor__max_depth': [3, 10, 20, 25, 50],
                              'GradientBoostingRegressor__n_estimators': [500, 800, 1000, 1200],
                              'GradientBoostingRegressor__subsample': [0.10, 0.5, 0.7, 1.0]}
            elif choice == 3:
                estimator = getRandomForestRegressor(best)
                param_grid = {'RandomForestRegressor__criterion': ['squared_error', 'absolute_error', 'poisson'],
                              'RandomForestRegressor__max_depth': [3, 10, 20, 25, 50],
                              'RandomForestRegressor__max_features': ['sqrt', 'log2', 2, 1, 0.5]}
            elif choice == 4:
                estimator = getDecisionTreeRegressor(best)
                param_grid = {}
            elif choice == 5:
                estimator = getExtraTreeRegressor(best)
                param_grid = {}
            else:
                print("\nPlease enter a valid choice!")

            if choice in [2, 3, 4, 5]:
                best_estimator = find_best_params(estimator, param_grid, X_train, y_train)
                model = ml.Model(config.model, model=best_estimator)
                model.model.fit(X_train, y_train)
                model.save()

                y_pred = model.model.predict(X_test)
                BSS.estimator.resultAnalysis(y_test, y_pred)
                BSS.estimator.plotPredictions(y_train, y_test, y_pred)


def findClassifierParams(dataset: Dataset, config: Config) -> None:
    """

    :param dataset: The loaded and processed dataset, should be a Dataset
    :param config:
    :return: None
    """
    # TODO: documentation
    dataset.apply(BSS.binaryEncode, dataset.target)

    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    best = False
    while True:
        print(f"""
            0 - Back
            1 - Use best params (toggle) - {best}
            2 - GradientBoostingClassifier
            3 - RandomForestClassifier
            4 - RidgeClassifier
            5 - DecisionTreeClassifier
            """)
        choice = input("Which classifier model: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        if choice is not None:
            classifier, param_grid = None, {}
            if choice == 0:
                return
            elif choice == 1:
                best = not best
            elif choice == 2:
                classifier = getGradientBoostingClassifier(best)
                param_grid = {'GradientBoostingClassifier__learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                              'GradientBoostingClassifier__max_depth': [3, 10, 20, 25, 50],
                              'GradientBoostingClassifier__n_estimators': [500, 800, 1000, 1200],
                              'GradientBoostingClassifier__subsample': [0.10, 0.5, 0.7, 1.0]}
            elif choice == 3:
                classifier = getRandomForestClassifier(best)
                param_grid = {'RandomForestClassifier__criterion': ['gini', 'entropy', 'log_loss'],
                              'RandomForestClassifier__max_depth': [3, 10, 20, 25, 50],
                              'RandomForestClassifier__max_features': ['sqrt', 'log2', 'auto', 1.0]}
            elif choice == 4:
                classifier = getRidgeClassifier(best)
                param_grid = {}
            elif choice == 5:
                classifier = getDecisionTreeClassifier(best)
                param_grid = {}
            else:
                print("\nPlease enter a valid choice!")

            if choice in [2, 3, 4, 5]:
                best_estimator = find_best_params(classifier, param_grid, X_train, y_train)
                model = ml.Model(config.model, model=best_estimator)
                model.model.fit(X_train, y_train)
                model.save()

                y_pred = model.model.predict(X_test)
                BSS.classifier.resultAnalysis(y_test, y_pred)
                BSS.classifier.plotClassifications(y_test, model.name, y_pred)
