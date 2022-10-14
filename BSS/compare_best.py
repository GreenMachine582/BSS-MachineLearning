from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from mlxtend.evaluate import bias_variance_decomp
from pandas import Series

import BSS
import machine_learning as ml
from machine_learning import Config, Dataset, Model, utils


def _plotBullseye(ax, x, y, title: str = ''):
    circle1 = plt.Circle((0, 0), 0.08, color='red', alpha=0.6)
    circle2 = plt.Circle((0, 0), 0.25, color='blue', linewidth=0.8, alpha=0.3, fill=False)
    circle3 = plt.Circle((0, 0), 0.55, color='red', linewidth=0.8, alpha=0.3, fill=False)
    circle4 = plt.Circle((0, 0), 0.85, color='blue', linewidth=0.8, alpha=0.3, fill=False)
    ax.add_artist(circle4)
    ax.add_artist(circle3)
    ax.add_artist(circle2)
    ax.add_artist(circle1)

    ax.set_aspect('equal')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(title)

    ax.scatter(x, y, c='g')
    ax.axis([-1, 1, -1, 1])
    return ax


def biasVarianceDecomp(X_train, X_test, y_train, y_test, n_iter: int = 10, display: bool = False,
                       dataset_name: str = '', dir_: str = ''):
    estimator = BSS.compare_params.getMLPRegressor()
    model = estimator['base'].set_params(**estimator['best_params'])
    loss, bias, var = [], [], []
    for i in range(n_iter):
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, X_train.values, y_train.values,
                                                                    X_test.values, y_test.values,
                                                                    num_rounds=5, loss='mse')
        if display:
            print('Average expected loss: %.3f' % avg_expected_loss)
            print('Average bias: %.3f' % avg_bias)
            print('Average variance: %.3f' % avg_var)
        loss.append(avg_expected_loss)
        bias.append(avg_bias)
        var.append(avg_var)
        logging.info(f"Bias variance iter: {i + 1}/{n_iter}")

    fig, ax = plt.subplots()
    _plotBullseye(ax, bias, var, title=estimator['fullname'])
    plt.show()


def compareEstimator(estimator, dataset, config):
    results_dir = ml.utils.makePath(config.dir_, config.results_folder, f"{estimator.type_}_{estimator.name}")

    X_train, X_test, y_train, y_test = dataset.split(shuffle=False)

    logging.info("Fitting and predicting")
    estimator.model.fit(X_train, y_train)
    y_pred = np.clip(estimator.model.predict(X_test), 0, None)

    estimator.save()

    estimator.resultAnalysis(y_test, y_pred, plot=False, dataset_name=dataset.name, dir_=results_dir)
    estimator.plotPrediction(y_test, Series(y_pred, index=y_test.index).resample('D').sum(), y_train,
                             target=dataset.target)

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

    classifier.resultAnalysis(y_test, y_pred, plot=False, dataset_name=dataset.name)
    classifier.plotPrediction(y_test, y_pred, target=dataset.target, dataset_name=dataset.name)
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
        8 - Bias Variance Decomposition (TBA)
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
            elif choice == 8:
                pass
                # X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)
                # biasVarianceDecomp(X_train, X_test, y_train, y_test)
            else:
                print("\nPlease enter a valid choice!")

        if model_config is not None:
            model = Model(config.model, **model_config)
            model.update(model=deepcopy(model.base).set_params(**model.best_params))
            if model.type_ == 'estimator':
                compareEstimator(model, dataset, config)
            elif model.type_ == 'classifier':
                compareClassifier(model, deepcopy(dataset), config)
