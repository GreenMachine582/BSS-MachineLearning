
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.cluster import SpectralBiclustering, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import BSS

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


def exploratoryDataAnalysis(dataset):
    logging.info("Exploratory Data Analysis")
    df = dataset.df

    # plots a corresponding matrix
    plt.figure()
    sn.heatmap(df.corr(), annot=True)


def extractFeatures(config, dataset):
    logging.info("Extracting features")
    df = dataset.df

    # adds some historical data
    df.loc[:, 'prev'] = df.loc[:, 'cnt'].shift()
    df.loc[:, 'diff'] = df.loc[:, 'prev'].diff()
    df.loc[:, 'prev-2'] = df.loc[:, 'prev'].shift()
    df.loc[:, 'diff-2'] = df.loc[:, 'prev-2'].diff()

    df = df.drop(['season', 'mnth'], axis=1)

    dataset.update(df=df, suffix='-extracted')
    dataset.handleMissingData()

    x = dataset.df.drop(config.target, axis=1)  # denotes independent features
    y = dataset.df[config.target]  # denotes dependent variables

    print(x.head())
    print(y.head())

    return dataset, x, y


def splitDataset(config, x, y):
    logging.info("Splitting data")

    size = round(x.shape[0] * config.split_ratio)
    x_train = x[:size]
    y_train = y[:size]
    x_test = x[size:]
    y_test = y[size:]

    return x_train, x_test, y_train, y_test


def compareModels(x_train, y_train):
    # TODO: separate the clustering techniques and apply appropriate prediction
    #  and scoring methods
    logging.info("Training model")
    models, names, results = [], [], []
    models.append(('LR', LinearRegression()))
    models.append(('NN', MLPRegressor(solver='lbfgs')))  # neural network
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('RF', RandomForestRegressor(n_estimators=10)))  # Ensemble method - collection of many decision trees
    models.append(('SVR', SVR(gamma='auto')))  # kernel = linear
    models.append(('GBR', GradientBoostingRegressor()))
    # models.append(('AC', AgglomerativeClustering(n_clusters=4)))
    # models.append(('BIC', SpectralBiclustering(n_clusters=(4, 3))))

    for name, model in models:
        tscv = TimeSeriesSplit(n_splits=10)  # TimeSeries Cross validation

        cv_results = cross_val_score(model, x_train, y_train, cv=tscv, scoring='r2')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # Compare Algorithms
    plt.figure()
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')


def trainModel(x_train, y_train):
    model = GradientBoostingRegressor()
    param_search = {'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
                    'max_depth': [2, 3, 4, 6, 8, 10, 12],
                    'n_estimators': [100, 300, 500, 700, 1000, 1500, 2000],
                    'subsample': [0.9, 0.5, 0.2, 0.1, 0.08, 0.06]}
    tscv = TimeSeriesSplit(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, n_jobs=-1)
    gsearch.fit(x_train, y_train)
    best_model = gsearch.best_estimator_
    best_score = gsearch.best_score_
    best_params = gsearch.best_params_
    return best_model, best_score, best_params


def main(dir_=local_dir):
    config = BSS.Config(dir_, dataset_name='Bike-Sharing-Dataset-day', model_technique='test',
                        model_algorithm='all')

    dataset = BSS.Dataset(config)

    if not dataset.load():
        logging.error("Couldn't load a dataset")

    dataset = processData(config, dataset)
    exploratoryDataAnalysis(dataset)

    dataset, x, y = extractFeatures(config, dataset)
    exploratoryDataAnalysis(dataset)

    x_train, x_test, y_train, y_test = splitDataset(config, x, y)

    compareModels(x_train, y_train)

    model, score, _ = trainModel(x_train, y_train)

    model = BSS.Model(config, model=model)
    model.save()
    model.resultAnalysis(score, x_test, y_test)

    plt.show()

    return


if __name__ == '__main__':
    main()
