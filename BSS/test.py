
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

    # plots a line graph of BSS Demand vs Date
    plt.figure()
    plt.plot(df.index, df['cnt'])
    plt.title('BSS Demand Vs Datetime')
    plt.xlabel('Datetime')
    plt.ylabel('Cnt')

    # TODO: Add graphs
    #  a) Bar graph - Demand vs temperature/weatheris (2 separate graphs)
    #  b) Box plots - Demand vs season/holiday/weekday/workingday (so 4 separate box plots)

    ### Write code below here ###



    plt.show()  # displays all figures


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
    models.append(('GBR', GradientBoostingRegressor(learning_rate=0.04, max_depth=2, n_estimators=1000, subsample=0.1)))
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
    param_search = {'learning_rate': [0.04],
                    'max_depth': [2],
                    'n_estimators': [1000],
                    'subsample': [0.1]}
    tscv = TimeSeriesSplit(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring='r2', n_jobs=-1, verbose=2)
    gsearch.fit(x_train, y_train)
    return gsearch.best_estimator_, gsearch.best_score_, gsearch.best_params_


def plotPredictions(model, dataset, x_test):
    # plots a line graph of BSS True and Predicted Demand vs Date
    predicted_demand = model.model.predict(x_test)
    plt.figure()
    plt.plot(dataset.df.index, dataset.df['cnt'], color='blue')
    plt.plot(x_test.index, predicted_demand, color='red')
    plt.title('BSS Demand Vs Datetime')
    plt.xlabel('Datetime')
    plt.ylabel('Cnt')
    plt.show()


def main(dir_=local_dir):
    config = BSS.Config(dir_, name='Bike-Sharing-Dataset-day')

    dataset = BSS.Dataset(config)

    if not dataset.load():
        logging.error("Couldn't load a dataset")

    dataset = processData(config, dataset)
    exploratoryDataAnalysis(dataset)

    dataset, x, y = extractFeatures(config, dataset)

    # plots a corresponding matrix
    plt.figure()
    sn.heatmap(dataset.df.corr(), annot=True)

    x_train, x_test, y_train, y_test = BSS.dataset.split(x, y, split_ratio=config.split_ratio)

    compareModels(x_train, y_train)

    plt.show()

    model, score, params = trainModel(x_train, y_train)

    print(params)

    model = BSS.Model(config, model=model)
    model.save()
    model.resultAnalysis(score, x_test, y_test)

    plotPredictions(model, dataset, x_test)

    return


if __name__ == '__main__':
    main()