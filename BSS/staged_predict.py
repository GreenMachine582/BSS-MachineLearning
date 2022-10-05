
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

import BSS
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def main(dataset, config):
    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    params = {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1000, 'subsample': 0.5}
    model = ml.Model(config.model, model=GradientBoostingRegressor(random_state=config.random_state, **params))

    model.model.fit(X_train, y_train)
    model.save()

    y_pred = model.model.predict(X_test)

    BSS.estimator.resultAnalysis(y_test, y_pred)

    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(model.model.staged_predict(X_test)):
        test_score[i] = model.model.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        model.model.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()

    feature_importance = model.model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(dataset.df.columns)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(model.model, X_test, y_test, n_repeats=10, random_state=config.random_state, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(dataset.df.columns)[sorted_idx])
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show()
