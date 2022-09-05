
import logging
import os

import BSS

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay,\
    mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression

# Constants
local_dir = os.path.dirname(__file__)


def processData(config, dataset):
    logging.info("Processing data")
    print(dataset.head())

    for i in range(dataset.shape[0]):
        label = dataset[config.target].iloc[i]
        dataset[config.target].iloc[i] = round(label / 100) * 100  # decrease the number of categories

    dataset[config.target] = dataset[config.target].astype('category')

    # dataset = dataset.drop(['instant', 'dteday', 'registered', 'casual'], axis=1)  # temp remove

    X = dataset.drop(config.target, axis=1)  # denotes independent features
    y = dataset[config.target]  # denotes dependent variables

    print(dataset.axes)
    print(dataset.head())

    print(dataset.isnull().sum())  # check for missing values
    print(dataset.dtypes)

    print("X shape:", X.shape)

    return dataset, X, y


def extractFeatures(dataset, x, y):
    logging.info("Extracting features")

    return dataset, x, y


def exploratoryDataAnalysis(dataset):
    logging.info("Exploratory Data Analysis")
    # plots a corresponding matrix
    plt.figure()
    sn.heatmap(dataset.corr(), annot=True)


def splitDataset(config, x, y):
    logging.info("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=config.split_ratio,
                                                        random_state=config.random_seed)
    return X_train, X_test, y_train, y_test


def trainModel(config, x_train, y_train):
    logging.info("Training model")
    model = None
    if config.model_type == "LogisticRegression":
        model = LogisticRegression(solver="lbfgs", max_iter=100)

    return model.fit(x_train, y_train)


def resultAnalysis(model, x_test, y_test):
    logging.info("Analysing results")
    score = model.score(x_test, y_test)
    print("Score - %.4f%s" % (score * 100, "%"))

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()

    # plt.show()  # displays all plots

    print(classification_report(y_test, y_pred))

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Mean Root Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))


def main():
    # using mnist_784 for testing
    config = BSS.Config(local_dir, 'mnist_784')  # 'Bike-Sharing-Dataset-day'

    raw_dataset = BSS.Dataset(config)

    if raw_dataset.dataset is None:
        logging.error("Couldn't load a dataset")

    dataset, X, y = processData(config, raw_dataset.dataset)

    dataset, X, y = extractFeatures(dataset, X, y)

    # exploratoryDataAnalysis(dataset)

    X_train, X_test, y_train, y_test = splitDataset(config, X, y)

    model = trainModel(config, X_train, y_train)

    resultAnalysis(model, X_test, y_test)

    return


if __name__ == '__main__':
    main()
