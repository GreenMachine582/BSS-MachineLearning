
import logging
import os

import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor

import BSS
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def plotPredictions(df: DataFrame, X_test: DataFrame, y_pred: ndarray) -> None:
    """
    Plots the BSS daily demand and predictions.

    :param df: the dataset itself, should be a DataFrame
    :param X_test: testing independent features, should be a DataFrame
    :param y_pred: predicted dependent variables, should be a ndarray
    :return: None
    """
    # plots a line graph of BSS True and Predicted Demand vs Date
    plt.figure()

    # Groups hourly instance into summed days, makes it easier to plot
    temp = DataFrame({'datetime': pd.to_datetime(df.index), 'cnt': df['cnt']})
    temp = temp.groupby(temp['datetime'].dt.date).sum()
    plt.plot(temp.index, temp['cnt'], color='blue')

    # Groups hourly instance into summed days, makes it easier to plot
    temp = DataFrame({'datetime': pd.to_datetime(X_test.index), 'y_pred': y_pred})
    temp = temp.groupby(temp['datetime'].dt.date).sum()
    plt.plot(temp.index, temp['y_pred'], color='red')

    plt.title('BSS Demand Vs Datetime')
    plt.xlabel('Datetime')
    plt.ylabel('Cnt')
    plt.show()


def main(dir_=local_dir):
    config = ml.Config(dir_, 'Bike-Sharing-Dataset-hour')

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        return

    dataset = BSS.processDataset(dataset)

    X, y = dataset.getIndependent(), dataset.getDependent()
    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    model = ml.Model(config.model, model=GradientBoostingRegressor(random_state=config.random_state))
    param_grid = {'loss': ['squared_error', 'absolute_error']}

    cv_results = model.gridSearch(param_grid, X, y)
    print('The best estimator:', cv_results.best_estimator_)
    print('The best score: %.2f%s' % (cv_results.best_score_ * 100, '%'))
    print('The best params:', cv_results.best_params_)

    model.save()

    y_pred = model.predict(X_test)
    ml.resultAnalysis(y_test, y_pred)
    plotPredictions(dataset.df, X_test, y_pred)

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
