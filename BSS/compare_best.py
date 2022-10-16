from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from pandas import Series

import BSS
import machine_learning as ml
from machine_learning import Config, Dataset, Model


def compareEstimator(estimator, dataset, config):
    results_dir = ml.utils.makePath(config.dir_, config.results_folder, f"{estimator.type_}_{estimator.name}")

    X_train, X_test, y_train, y_test = dataset.split(shuffle=False)

    logging.info("Fitting and predicting")
    estimator.model.fit(X_train, y_train)
    y_pred = np.clip(estimator.model.predict(X_test), 0, None)

    estimator.save()

    estimator.resultAnalysis(y_test, y_pred, plot=False, dataset_name=f"{dataset.name} Recorded Best")
    estimator.plotPrediction(y_test.resample('D').sum(), Series(y_pred, index=y_test.index).resample('D').sum(),
                             y_train.resample('D').sum(), target=dataset.target,
                             dataset_name=f"{dataset.name} Recorded Best", dir_=results_dir)
    estimator.plotImportance(dataset.df.columns, X_test, y_test, dataset_name=dataset.name, dir_=results_dir)


def compareClassifier(classifier, dataset, config):
    results_dir = ml.utils.makePath(config.dir_, config.results_folder, f"{classifier.type_}_{classifier.name}")

    dataset.df.drop('diff', axis=1, inplace=True)  # same feature as binary encoded target
    dataset.apply(ml.binaryEncode, dataset.target)

    X_train, X_test, y_train, y_test = dataset.split(shuffle=False)

    logging.info("Fitting and predicting")
    classifier.model.fit(X_train, y_train)
    y_pred = classifier.model.predict(X_test)

    classifier.save()

    classifier.resultAnalysis(y_test, y_pred, plot=False, dataset_name=f"{dataset.name} Recorded Best")
    classifier.plotImportance(dataset.df.columns, X_test, y_test, dataset_name=dataset.name, dir_=results_dir)


def compareBest(dataset: Dataset, config: Config) -> None:
    """

    :param dataset: The loaded and processed dataset, should be a Dataset
    :param config:
    :return: None
    """
    # TODO: documentation

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
        choice = input("Which option number: ")
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
                model_config = BSS.compare_params.getGradientBoostingRegressor()
            elif choice == 2:
                model_config = BSS.compare_params.getRandomForestRegressor()
            elif choice == 3:
                model_config = BSS.compare_params.getMLPRegressor()
            elif choice == 4:
                model_config = BSS.compare_params.getGradientBoostingClassifier()
            elif choice == 5:
                model_config = BSS.compare_params.getRandomForestClassifier()
            elif choice == 6:
                model_config = BSS.compare_params.getRidgeClassifier()
            elif choice == 7:
                model_config = BSS.compare_params.getMLPClassifier()
            else:
                print("\nPlease enter a valid choice!")

        if model_config is not None:
            model = Model(config.model, **model_config)
            model.update(model=deepcopy(model.base).set_params(**model.best_params))
            if model.type_ == 'estimator':
                compareEstimator(model, dataset, config)
            elif model.type_ == 'classifier':
                compareClassifier(model, deepcopy(dataset), config)
