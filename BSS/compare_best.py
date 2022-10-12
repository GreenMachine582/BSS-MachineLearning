from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from mlxtend.evaluate import bias_variance_decomp

import BSS
from machine_learning import Config, Dataset, Model


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


def biasVarianceDecomp(X_train, X_test, y_train, y_test, n_iter: int = 10, show: bool = False):
    estimator = BSS.compare_params.getMLPRegressor()
    model = estimator['base'].set_params(**estimator['best_params'])
    loss, bias, var = [], [], []
    for i in range(n_iter):
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, X_train.values, y_train.values,
                                                                    X_test.values, y_test.values,
                                                                    num_rounds=5, loss='mse')
        if show:
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


def compareBestEstimators(dataset: Dataset, config: Config) -> None:
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
        Compare predictions
            1 - Gradient Boosting Regressor
            2 - Random Forest Regressor
            3 - K-Neighbors Regressor
            4 - MLP Regressor
        5 - Bias Variance Decomposition
        """)
        choice = input("Which option number: ")
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
                estimator = BSS.compare_params.getGradientBoostingRegressor()
            elif choice == 2:
                estimator = BSS.compare_params.getRandomForestRegressor()
            elif choice == 3:
                estimator = BSS.compare_params.getKNeighborsRegressor()
            elif choice == 4:
                estimator = BSS.compare_params.getMLPRegressor()
            elif choice == 5:
                biasVarianceDecomp(X_train, X_test, y_train, y_test)
            else:
                print("\nPlease enter a valid choice!")

        if estimator is not None:
            estimator = Model(config.model, **estimator)

            estimator.update(model=deepcopy(estimator.base))
            estimator.model.set_params(**estimator.best_params)

            logging.info("Fitting and predicting")
            estimator.model.fit(X_train, y_train)
            y_pred = np.clip(estimator.model.predict(X_test), 0, None)

            estimator.resultAnalysis(y_test, y_pred)
            estimator.plotPrediction(y_train, y_test, y_pred, target=dataset.target)
