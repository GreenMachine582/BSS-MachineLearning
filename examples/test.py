
import logging
import math
import os

import BSS

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, KFold, GridSearchCV
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Constants
local_dir = os.path.dirname(__file__)


def processData(config, dataset):
    logging.info("Processing data")
    print(dataset.head())

    # for i in range(dataset.shape[0]):
    #     label = dataset[config.target].iloc[i]
    #     dataset[config.target].iloc[i] = math.ceil(label / 10) * 10  # decrease the number of categories

    # dataset[config.target] = dataset[config.target].astype('category')

    dataset['dteday'] = pd.to_datetime(dataset['dteday'])
    dataset.index = dataset['dteday']

    dataset = dataset.drop(['instant', 'dteday', 'registered', 'casual'], axis=1)  # temp remove

    # dataset = dataset[['cnt']]
    # dataset.loc[:, 'prev'] = dataset.loc[:, 'cnt'].shift()
    # dataset.loc[:, 'diff'] = dataset.loc[:, 'prev'].diff()

    dataset = dataset.dropna()

    X = dataset.drop(config.target, axis=1)  # denotes independent features
    y = dataset[config.target]  # denotes dependent variables

    print(dataset.axes)
    print(dataset.head())

    print(dataset.isnull().sum())  # check for missing values
    print(dataset.dtypes)

    print("X shape:", X.shape)

    return dataset, X, y


def exploratoryDataAnalysis(dataset, x, y):
    logging.info("Exploratory Data Analysis")
    # plots a corresponding matrix
    plt.figure()
    sn.heatmap(dataset.corr(), annot=True)

    # plots a bar graph to represent number of instances per target/label
    # plt.figure()
    # y.value_counts().plot(kind="bar")

    # plt.show()


def extractFeatures(dataset, x, y):
    logging.info("Extracting features")

    return dataset, x, y


def splitDataset(config, x, y):
    logging.info("Splitting data")

    # X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=config.split_ratio,
    #                                                     random_state=config.random_seed)

    size = round(x.shape[0] * config.split_ratio)
    X_train = x[:size]
    y_train = y[:size]
    X_test = x[size:]
    y_test = y[size:]

    return X_train, X_test, y_train, y_test


def compareModels(x_train, y_train):
    logging.info("Training model")
    models, names, results = [], [], []
    models.append(('LR', LinearRegression()))
    models.append(('NN', MLPRegressor(solver='lbfgs')))  # neural network
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('RF', RandomForestRegressor(n_estimators=10)))  # Ensemble method - collection of many decision trees
    models.append(('SVR', SVR(gamma='auto')))  # kernel = linear

    for name, model in models:
        tscv = TimeSeriesSplit(n_splits=10)  # TimeSeries Cross validation
        kfold = KFold(n_splits=5)

        cv_results = cross_val_score(model, x_train, y_train, cv=tscv, scoring='r2')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # Compare Algorithms
    plt.figure()
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()


def trainModel(x_train, y_train):
    rmse_score = metrics.make_scorer(rmse, greater_is_better=False)
    model = RandomForestRegressor()
    param_search = {
        'n_estimators': [20, 50, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [i for i in range(5, 15)]
    }
    tscv = TimeSeriesSplit(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring=rmse_score)
    gsearch.fit(x_train, y_train)
    best_model = gsearch.best_estimator_
    best_score = gsearch.best_score_
    return best_model, best_score


def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score


def regression_results(y_test, y_pred):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_test, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    print('explained_variance: ', round(explained_variance, 4))
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))


def resultAnalysis(model, score, x_train, y_train, x_test, y_test):
    logging.info("Analysing results")

    print("Score - %.4f%s" % (score * 100, "%"))

    y_pred = model.predict(x_test)

    regression_results(y_test, y_pred)


def main(dir_=''):
    config = BSS.Config(dir_, dataset_name='Bike-Sharing-Dataset-day', model_technique='test',
                        model_algorithm='all')

    raw_dataset = BSS.Dataset(config)
    raw_dataset.load()

    if raw_dataset.dataset is None:
        logging.error("Couldn't load a dataset")

    dataset, X, y = processData(config, raw_dataset.dataset)

    exploratoryDataAnalysis(dataset, X, y)

    dataset, X, y = extractFeatures(dataset, X, y)

    X_train, X_test, y_train, y_test = splitDataset(config, X, y)

    compareModels(X_train, y_train)

    model, score = trainModel(X_train, y_train)

    BSS.Model(config, model=model).save()

    resultAnalysis(model, score, X_train, y_train, X_test, y_test)

    return


if __name__ == '__main__':
    main(local_dir)
