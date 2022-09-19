
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.cluster import SpectralBiclustering, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import BSS

# Constants
local_dir = os.path.dirname(__file__)


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
    #   See the graphs on pg. 18 as reference.
    #   https://www.researchgate.net/publication/337062461_Regression_Model_for_Bike-Sharing_Service_by_Using_Machine_Learning
    #  a) Bar graph - Demand vs temperature/weatheris (2 separate graphs)
    #  b) Box plots - Demand vs season/holiday/weekday/workingday (so 4 separate box plots)

    ### Write code below here ###



    plt.show()  # displays all figures


def extractFeatures(config, dataset):
    logging.info("Extracting features")
    df = dataset.df

    df = df.drop(['yr', 'mnth'], axis=1)

    dataset.update(df=df, suffix='-extracted')
    dataset.handleMissingData()

    x = dataset.df.drop(config.target, axis=1)  # denotes independent features
    y = dataset.df[config.target]  # denotes dependent variables

    print(x.head())
    print(y.head())

    return dataset


def compareModels(x_train, y_train):
    # TODO: separate the clustering techniques and apply appropriate prediction
    #  and scoring methods
    logging.info("Comparing models")
    models, names, results = [], [], []
    # models.append(('LR', LinearRegression()))
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

    plt.figure()
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')


def trainModel(x_train, y_train):
    logging.info("Training best model")
    model = GradientBoostingRegressor(learning_rate=0.09, max_depth=6, n_estimators=600, subsample=0.12)
    model.fit(x_train, y_train)
    return model


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
    config = BSS.Config(dir_, name='Bike-Sharing-Dataset-day', suffix='-pre-processed')

    dataset = BSS.Dataset(config)
    # Checks if BSS dataset was loaded
    if dataset.df is None:
        logging.warning(f"'DataFrame' object was expected, got {type(dataset.df)}")
        return

    # Process the dataset
    dataset.apply(BSS.process.processData)
    dataset.update(suffix='-processed')

    exploratoryDataAnalysis(dataset)

    # dataset = extractFeatures(config, dataset)

    x, y = dataset.split()

    # plots a corresponding matrix
    plt.figure()
    sn.heatmap(dataset.df.corr(), annot=True)

    compareModels(x['train'], y['train'])

    plt.show()

    model = trainModel(x['train'], y['train'])
    score = model.score(x['test'], y['test'])

    model = BSS.Model(config, model=model)
    model.save()
    model.resultAnalysis(score, x['test'], y['test'])

    plotPredictions(model, dataset, x['test'])

    return


if __name__ == '__main__':
    main()
