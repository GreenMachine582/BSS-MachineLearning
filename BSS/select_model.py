from __future__ import annotations

import logging
import math
import os

from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering, SpectralBiclustering
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import BSS

# Constants
local_dir = os.path.dirname(__file__)


def convertToCategorical(df: DataFrame, future_categories: int = 1, num_of_categories: int = 20) -> DataFrame:
    # TODO: Documentation
    if future_categories > num_of_categories - 1:
        logging.warning(f"Number_of_categories must be larger than future_categories")
        return df

    interval = math.floor(max(df['cnt']) / ((num_of_categories - 1) - future_categories))

    def temp(x):
        for i in range(num_of_categories):
            if x < i * interval:
                return i * interval

    df['cnt'] = df['cnt'].apply(temp)
    df['cnt'] = df['cnt'].astype("category")
    return df


def getContinuousModels(random_seed: int = None) -> list:
    models = [('LR', LinearRegression()),
              ('NN', MLPRegressor(max_iter=1200, random_state=random_seed)),
              ('KNN', KNeighborsRegressor()),
              ('RF', RandomForestRegressor(random_state=random_seed)),
              ('SVR', SVR()),
              ('GBR', GradientBoostingRegressor(random_state=random_seed))]
    return models


def getCategoricalModels(random_seed: int = None) -> list:
    models = [('LR', LogisticRegression()),]
              # ('AC', AgglomerativeClustering()),
              # ('BIC', SpectralBiclustering(random_state=random_seed))]
    return models


def compareModels(X_train, y_train, models, scoring: str = None, folds: int = 10, time_series: bool = False):
    """
    Trains and cross validates a basic model with default params.

    :param X_train: training independent features, should be a DataFrame
    :param y_train: training dependent variables, should be a DataFrame
    :param models:
    :param scoring:
    :param folds: number of folds for cross-validation, should be an int
    :param time_series: whether the dataset is a time series, should be a bool
    :return: cv_results - ndarray[float]
    """
    # TODO: Documentation
    logging.info("Comparing models")
    names, results = [], []

    for name, model in models:
        cv = folds  # n Folds
        if time_series:
            cv = TimeSeriesSplit(n_splits=folds)  # TimeSeries Cross validation
        cv_results = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv)
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    return results, names


def plotPredictions(model, dataset, X_test):
    # TODO: Fix, Change, remove?
    # plots a line graph of BSS True and Predicted Demand vs Date
    predicted_demand = model.model.predict(X_test)
    plt.figure()
    plt.plot(dataset.df.index, dataset.df['cnt'], color='blue')
    plt.plot(X_test.index, predicted_demand, color='red')
    plt.title('BSS Demand Vs Datetime')
    plt.xlabel('Datetime')
    plt.ylabel('Cnt')
    plt.show()


def main(dir_: str = local_dir) -> None:
    """


    :param dir_: project's path directory, should be a str
    :return: None
    """
    # TODO: Documentation and error handling
    config = BSS.Config(dir_, 'Bike-Sharing-Dataset-day')

    dataset = BSS.Dataset(config.dataset)
    if not dataset.load():
        return

    dataset = BSS.processDataset(dataset)

    logging.info('Selecting Dependent Feature Type')
    while True:
        print("""
            0 - Back
            1 - Continuous
            2 - Categorical
            """)
        choice = input("Which question number: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        if choice is not None:
            if choice == 0:
                return
            elif choice == 1:
                models = getContinuousModels(config.random_seed)
                break
            elif choice == 2:
                dataset.apply(convertToCategorical)
                models = getCategoricalModels(config.random_seed)
                break
            else:
                print("\nPlease enter a valid choice!")

    X_train, X_test, y_train, y_test = dataset.split(config.random_seed)

    results, names = compareModels(X_train, y_train, models)
    plt.figure()
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

    # Remove models that performed poorly
    if choice == 1:
        del results[4]
        del results[2]
        del names[4]
        del names[2]

        plt.figure()
        plt.boxplot(results, labels=names)
        plt.title('Algorithm Comparison')
        plt.show()
    else:
        pass

    # Train best model
    if choice == 1:
        model = RandomForestRegressor(random_state=config.random_seed)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        model = BSS.Model(config.model, model=model)
        model.save()
        model.resultAnalysis(score, X_test, y_test)

        plotPredictions(model, dataset, X_test)
    else:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = accuracy_score(y_test, y_pred)
        print("Accuracy - %.4f%s" % (score * 100, "%"))

        model = BSS.Model(config.model, model=model)
        model.save()

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
