import logging
import os

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import BSS
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def biasVarianceDecomp(model, X_train, X_test, y_train, y_test):
    n_repeat = 20  # Number of iterations

    y_test = np.array(y_test)
    all_pred = np.zeros((n_repeat, y_test.size))
    for j in range(n_repeat):
        all_pred[j] = model.fit(X_train, y_train).predict(X_test)

    avg_preds = np.mean(all_pred, axis=0)
    avg_expected_loss = np.apply_along_axis(lambda x: ((x - y_test) ** 2).mean(), axis=1, arr=all_pred).mean()
    avg_bias = np.sum((avg_preds - y_test) ** 2) / y_test.size
    avg_var = np.sum((avg_preds - all_pred) ** 2) / all_pred.size
    return avg_expected_loss, avg_bias, avg_var


# def bullseyePlot(bias, variance):
#     fig, ax = plt.subplots()
#
#     max_value = max(max(bias), max(variance))
#     max_value += max_value * 0.10
#     circle1 = plt.Circle((0, 0), max_value * 0.3, color='green', alpha=0.3)
#     circle2 = plt.Circle((0, 0), max_value * 0.6, color='yellow', alpha=0.3)
#     circle3 = plt.Circle((0, 0), max_value * 0.9, color='red', alpha=0.3)
#
#     ax.add_artist(circle3)
#     ax.add_artist(circle2)
#     ax.add_artist(circle1)
#
#     plt.scatter(bias, variance)
#     plt.axis([-max_value, max_value, -max_value, max_value])
#
#     plt.show()


def main(dir_=local_dir):
    config = ml.Config(dir_, 'Bike-Sharing-Dataset-hour')
    # config = ml.Config(dir_, 'london_merged-hour')

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        return

    dataset = BSS.processDataset(dataset)

    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    model = ml.Model(config.model, model=GradientBoostingRegressor(random_state=config.random_state))
    param_grid = {'learning_rate': [0.01], 'max_depth': [10], 'n_estimators': [1000], 'subsample': [0.5]}

    cv_results = model.gridSearch(param_grid, X_train, y_train)
    print('The best estimator:', cv_results.best_estimator_)
    print('The best score: %.2f%s' % (cv_results.best_score_ * 100, '%'))
    print('The best params:', cv_results.best_params_)

    error, bias, var = biasVarianceDecomp(GradientBoostingRegressor(**cv_results.best_params_),
                                          X_train, X_test, y_train, y_test)

    print('Error: %.4f' % error)
    print('Bias: %.4f' % bias)
    print('Variance: %.4f' % var)

    model.update(model=cv_results.best_estimator_)
    model.model.fit(X_train, y_train)
    model.save()

    y_pred = model.model.predict(X_test)
    BSS.estimator.resultAnalysis(y_test, y_pred)
    BSS.estimator.plotPredictions(y_train, y_test, y_pred)

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
