import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

import BSS
from machine_learning import Dataset, Config, Model

# Constants
local_dir = os.path.dirname(__file__)


def stagedPredict(model, param, X_test, y_test):
    logging.info(f"Staged predict {model.name}")
    errors = np.zeros(param, dtype=np.float64)
    for i, y_pred in enumerate(model.model.staged_predict(X_test)):
        errors[i] = mean_squared_error(y_test, y_pred)
    return errors


def partialFit(model, param, X_train, X_test, y_train, y_test):
    logging.info(f"Partial fit {model.name}")
    errors = np.zeros(param, dtype=np.float64)
    for i in range(param):
        model.model.partial_fit(X_train, y_train)
        y_pred = model.model.predict(X_test)
        errors[i] = mean_squared_error(y_test, y_pred)
    return model, errors


def plotFeatureImportance(model, names, X_test, y_test):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(names)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(model, X_test, y_test, n_repeats=10, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(names)[sorted_idx])
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show()


def main(dataset: Dataset, config: Config) -> None:
    """

    :param dataset: BSS dataset, should be a Dataset
    :param config: BSS configuration, should be a Config
    :return: None
    """
    X_train, X_test, y_train, y_test = dataset.split(shuffle=False)

    estimator = Model(config.model, **BSS.compare_params.getMLPRegressor())
    estimator.createModel(param_type='best')

    estimator, errors = partialFit(estimator, estimator.best_params['max_iter'], X_train, X_test, y_train, y_test)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(np.arange(estimator.best_params["max_iter"]) + 1, estimator.model.train_score_, "b-",
             label="Training Set Deviance")
    plt.plot(np.arange(estimator.best_params["max_iter"]) + 1, errors, "r-",
             label="Test Set Deviance")
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()

    estimator.model.fit(X_train, y_train)
    plotFeatureImportance(estimator.model, dataset.df.columns, X_test, y_test)

    # part2
    estimator = Model(config.model, **BSS.compare_params.getGradientBoostingRegressor())
    estimator.createModel(param_type='best')

    estimator.model.fit(X_train, y_train)
    errors = stagedPredict(estimator, estimator.best_params['n_estimators'], X_test, y_test)
    plotFeatureImportance(estimator.model, dataset.df.columns, X_test, y_test)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(np.arange(estimator.best_params["n_estimators"]) + 1, estimator.model.train_score_, "b-",
             label="Training Set Deviance")
    plt.plot(np.arange(estimator.best_params["n_estimators"]) + 1, errors, "r-",
             label="Test Set Deviance")
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()

    plotFeatureImportance(estimator.model, dataset.df.columns, X_test, y_test)
