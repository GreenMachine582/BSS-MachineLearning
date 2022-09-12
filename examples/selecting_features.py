from __future__ import annotations

import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import BSS
from examples import test

# Constants
local_dir = os.path.dirname(__file__)


def processData(config, dataset):
    logging.info("Processing data")
    df = dataset.df

    print(df.head())

    df['dteday'] = pd.to_datetime(df['dteday'])
    df.index = df['dteday']

    df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

    dataset.update(df=df, suffix='-processed')
    dataset.handleMissingData()

    df = dataset.df

    print(df.axes)
    print(df.head())
    print(df.dtypes)

    x = df.drop(config.target, axis=1)  # denotes independent features
    print("X shape:", x.shape)

    return dataset


def extractFeatures(config, dataset):
    logging.info("Extracting features")
    df = dataset.df

    # adds some historical data
    df.loc[:, 'prev'] = df.loc[:, 'cnt'].shift()
    df.loc[:, 'diff'] = df.loc[:, 'prev'].diff()
    df.loc[:, 'prev-2'] = df.loc[:, 'prev'].shift()
    df.loc[:, 'diff-2'] = df.loc[:, 'prev-2'].diff()

    dataset.update(df=df, suffix='-extracted')
    dataset.handleMissingData()

    x = dataset.df.drop(config.target, axis=1)  # denotes independent features
    y = dataset.df[config.target]  # denotes dependent variables

    print(x.head())
    print(y.head())

    return dataset, x, y


def compareModels(x_train, y_train):
    # TODO: separate the clustering techniques and apply appropriate prediction
    #  and scoring methods
    logging.info("Training model")
    models, names, results = [], [], []
    models.append(('LR', LinearRegression()))
    models.append(('NN', MLPRegressor(solver='lbfgs')))  # neural network
    models.append(('RF', RandomForestRegressor(n_estimators=10)))  # Ensemble method - collection of many decision trees
    models.append(('GBR', GradientBoostingRegressor()))

    scores = np.zeros((4, 2))  # 4 for four active models in compareModels
    i = 0
    for name, model in models:
        tscv = TimeSeriesSplit(n_splits=10)  # TimeSeries Cross validation

        cv_results = cross_val_score(model, x_train, y_train, cv=tscv, scoring='r2')
        results.append(cv_results)
        names.append(name)
        scores[i] = cv_results.mean(), cv_results.std()
        i += 1
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # Compare Algorithms
    plt.figure()
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    return scores


def main(dir_=local_dir):
    config = BSS.Config(dir_, dataset_name='Bike-Sharing-Dataset-day', model_technique='test',
                        model_algorithm='all')

    dataset = BSS.Dataset(config)
    if not dataset.load():
        logging.error("Couldn't load a dataset")
    dataset = processData(config, dataset)

    # edit these values
    num_of_runs = 5
    selected_features = [[], ['season'], ['yr'], ['mnth'], ['holiday'], ['weekday'], ['workingday'],
                         ['season', 'mnth'], ['weekday', 'workingday']]

    results = []
    for features in selected_features:
        temp_dataset = deepcopy(dataset)
        temp_dataset.df = temp_dataset.df.drop(features, axis=1)
        temp_dataset, x, y = extractFeatures(config, temp_dataset)
        x_train, x_test, y_train, y_test = test.splitDataset(config, x, y)

        scores = np.zeros((4, 2))  # 4 for four active models in compareModels
        for i in range(num_of_runs):
            scores = np.add(scores, compareModels(x_train, y_train))
        scores = scores / num_of_runs  # average scores
        results.append(scores)

    for features, scores in zip(selected_features, results):
        print(features)
        print(scores)


if __name__ == '__main__':
    main()
