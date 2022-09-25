
import logging
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.cluster import SpectralBiclustering, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import BSS
sns.set(style='darkgrid')
df = pd.read_csv("day.csv")

# Constants
local_dir = os.path.dirname(__file__)


def exploratoryDataAnalysis(dataset):
    # TODO: Fix, Change, remove?
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
    #  a) Bar graph - Demand vs atemp/weather_code (2 separate graphs)
    #  b) Box plots - Demand vs season/is_holiday/is_weekend (so 3 separate box plots)

    ### Write code below here ###
def Box_plot(x_label, y_label):
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x=x_label, y=y_label, data=df)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    plt.show()


def bar_graph(x, y, x_label, y_label,type=0):
    if type:
        plt.bar(x, y, align='center', tick_label=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8","0.9","1"])
    else:
        plt.bar(x, y,tick_label=["1","2","3"], align='center')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def weathersit_data_process():
    cnt = df['cnt']
    weathersit = df['weathersit']
    x = [1, 2, 3]
    y = [0, 0, 0]
    for i in range(len(weathersit)):
        if weathersit[i] == 1:
            y[0] = y[0] + cnt[i]
        elif weathersit[i] == 2:
            y[1] = y[1] + cnt[i]
        elif weathersit[i] == 3:
            y[2] = y[2] + cnt[i]
    return x, y


def temperature_data_process():
    cnt = df['cnt']
    temp = df['temp']
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(temp)):
        if temp[i] <= 0.1:
            y[0] = y[0] + cnt[i]
        elif 0.1 < temp[i] <= 0.2:
            y[1] = y[1] + cnt[i]
        elif 0.2 < temp[i] <= 0.3:
            y[2] = y[2] + cnt[i]
        elif 0.3 < temp[i] <= 0.4:
            y[3] = y[3] + cnt[i]
        elif 0.4 < temp[i] <= 0.5:
            y[4] = y[4] + cnt[i]
        elif 0.5 < temp[i] <= 0.6:
            y[5] = y[5] + cnt[i]
        elif 0.6 < temp[i] <= 0.7:
            y[6] = y[6] + cnt[i]
        elif 0.7 < temp[i] <= 0.8:
            y[7] = y[7] + cnt[i]
        elif 0.8 < temp[i] <= 0.9:
            y[8] = y[8] + cnt[i]
        elif 0.9 < temp[i] <= 1:
            y[9] = y[9] + cnt[i]
    return x, y

def main1():
    Box_plot('season','cnt')
    Box_plot('holiday','cnt')
    Box_plot('weekday','cnt')
    Box_plot('workingday','cnt')
    x1, y1 = temperature_data_process()
    bar_graph(x1, y1, 'temperature', 'cnt',1)
    x2, y2 = weathersit_data_process()
    bar_graph(x2, y2, 'weathersit', 'cnt')


    plt.show()  # displays all figures


def extractFeatures(df, target: str):
    # TODO: Fix, Change, remove?
    logging.info("Extracting features")

    df = BSS.handleMissingData(df)

    # x = df.drop(target, axis=1)  # denotes independent features
    # y = df[target]  # denotes dependent variables
    #
    # print(x.head())
    # print(y.head())

    return df


def compareModels(x_train, y_train, random_seed: int = None):
    # TODO: separate the clustering techniques and apply appropriate prediction
    #  and scoring methods
    logging.info("Comparing models")
    models, names, results = [], [], []
    # models.append(('LR', LinearRegression()))
    models.append(('NN', MLPRegressor(solver='lbfgs', random_state=random_seed)))  # neural network
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('RF', RandomForestRegressor(n_estimators=10, random_state=random_seed)))  # Ensemble method - collection of many decision trees
    # models.append(('SVR', SVR(gamma='auto')))  # kernel = linear
    models.append(('GBR', GradientBoostingRegressor(random_state=random_seed)))
    # models.append(('AC', AgglomerativeClustering(n_clusters=4)))
    # models.append(('BIC', SpectralBiclustering(n_clusters=(4, 3))))

    for name, model in models:
        tscv = TimeSeriesSplit(n_splits=10)  # TimeSeries Cross validation

        cv_results = cross_val_score(model, x_train, y_train, cv=tscv)
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # plt.figure()
    # plt.boxplot(results, labels=names)
    # plt.title('Algorithm Comparison')


def trainModel(x_train, y_train, random_seed: int = None):
    # TODO: Fix, Change, remove?
    logging.info("Training best model")
    model = GradientBoostingRegressor(learning_rate=0.09, max_depth=6, n_estimators=600, subsample=0.12,
                                      random_state=random_seed)
    model.fit(x_train, y_train)
    return model


def plotPredictions(model, dataset, x_test):
    # TODO: Fix, Change, remove?
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
    # TODO: Fix, Change, remove?
    config = BSS.Config(dir_, 'Bike-Sharing-Dataset-hour')

    dataset = BSS.Dataset(config.dataset)
    if not dataset.load():
        return

    dataset = BSS.processDataset(dataset)

    # exploratoryDataAnalysis(dataset)

    # dataset.apply(extractFeatures, dataset.target)
    # dataset.update(name=config.name + '-extracted')

    X_train, X_test, y_train, y_test = dataset.split(config.random_seed)

    compareModels(X_train, y_train, random_seed=config.random_seed)

    plt.show()

    model = trainModel(X_train, y_train, random_seed=config.random_seed)
    score = model.score(X_test, y_test)

    model = BSS.Model(config.model, model=model)
    model.save()
    model.resultAnalysis(score, X_test, y_test)

    # plotPredictions(model, dataset, X_test)

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
    main1()
