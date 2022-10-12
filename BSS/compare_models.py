from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace
from pandas import DataFrame, Series
from seaborn import heatmap
from sklearn import ensemble, neighbors, neural_network, svm, tree, linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_validate

import BSS
from machine_learning import Dataset


def compareModels(models: dict, X_train: DataFrame, y_train: Series, dataset_name: str = '', dir_: str = '') -> dict:
    """
    Cross validates each model with a time series split and plots comparison
    graphs of test scores and fitting times.

    :param models: The models to be compared, should be a dict[str: Any]
    :param X_train: Training independent features, should be a DataFrame
    :param y_train: Training dependent variables, should be a Series
    :param dataset_name: Name of dataset, should be a str
    :param dir_: Save location for figures and results, should be a str
    :return: results - dict[str: Any]
    """
    results = {}
    for name in models:
        cv_results = cross_validate(models[name], X_train, y_train, cv=TimeSeriesSplit(10), n_jobs=-1)
        cv_results['model'] = models[name]
        results[name] = cv_results

        print('%s: %f (%f)' % (name, cv_results['test_score'].mean(), cv_results['test_score'].std()))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
    ax1.boxplot([results[name]['test_score'] for name in results], labels=[name for name in results])
    ax1.set(ylabel="Testing Score")
    ax2.boxplot([results[name]['fit_time'] for name in results], labels=[name for name in results])
    ax2.set(ylabel="Fitting Time")
    fig.suptitle(f"Model Comparison - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
    plt.show()
    return results


def compareEstimators(dataset: Dataset, config: Config) -> None:
    """
    Cross validates the estimators with a time series split then compares
    fitting times, test scores, results analyses and plots predicted
    estimations.

    :param dataset: BSS dataset, should be a Dataset
    :param random_state: Controls the random seed, should be an int
    :return: None
    """
    estimators = {'GBR': ensemble.GradientBoostingRegressor(random_state=random_state),
                  'RFR': ensemble.RandomForestRegressor(random_state=random_state),
                  'KNR': neighbors.KNeighborsRegressor(),
                  'MLPR': neural_network.MLPRegressor(max_iter=800, random_state=random_state),
                  'SVR': svm.SVR(),
                  'DTR': tree.DecisionTreeRegressor(random_state=random_state),
                  'ETR': tree.ExtraTreeRegressor(random_state=random_state)}

    X_train, X_test, y_train, y_test = dataset.split(random_state=random_state, shuffle=False)

    results_dir = utils.makePath(config.dir_, config.results_folder, 'compare_estimators')

    results = compareModels(estimators, X_train, y_train, dataset_name=dataset.name, dir_=results_dir)

    # removes estimators that performed poorly
    del results['KNR']
    del results['SVR']
    del results['ETR']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
    ax1.boxplot([results[name]['test_score'] for name in results], labels=[name for name in results])
    ax1.set(ylabel="Testing Score")
    ax2.boxplot([results[name]['fit_time'] for name in results], labels=[name for name in results])
    ax2.set(ylabel="Fitting Time")
    fig.suptitle(f"Model Comparison (Closeup) - {dataset.name}")
    if results_dir:
        plt.savefig(utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))
    plt.show()

    logging.info("Fitting and predicting")
    predictions = []
    for name in results:
        results[name]['model'].fit(X_train, y_train)
        y_pred = results[name]['model'].predict(X_test)
        predictions.append((name, np.clip(y_pred, 0, None)))

    plotEstimatorResultAnalysis(y_test, predictions)

    plotPredictions(y_train, y_test, predictions)


def plotClassifierResultAnalysis(y_test: Series, predictions: list) -> None:
    """
    Plots the results analysis bar graph for each classifier.

    :param y_test: Testing dependent variables, should be a Series
    :param predictions: Classifiers predictions, should be a list[tuple[str, ndarray]]
    :return: None
    """
    results, labels = {}, []
    for name, y_pred in predictions:
        results[name] = BSS.classifier.resultAnalysis(y_test, y_pred, show=False)
        labels.append(name)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8, 4), sharex='row')
    fig.suptitle('Result Analysis')
    _plotBar(ax1, labels, [results[name]['accuracy'] for name in results], 'Accuracy')
    _plotBar(ax2, labels, [results[name]['precision'] for name in results], 'Precision')
    _plotBar(ax3, labels, [results[name]['recall'] for name in results], 'Recall')
    _plotBar(ax4, labels, [results[name]['f1'] for name in results], 'F1')
    plt.show()


def plotClassifications(y_test: Series, names: list, predictions: list) -> None:
    """
    Plots confusion matrices to compare the classifiers predictions.

    :param y_test: Testing dependent variables, should be a Series
    :param names: The classifiers names, should be a list[str]
    :param predictions: The classifiers predictions, should be a list[ndarray]
    :return: None
    """
    if len(predictions) != 4:
        logging.warning(f"Incorrect number of names and predictions")
        return

    # Heatmaps of confusion matrices
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    cm = confusion_matrix(y_test, predictions[0])
    heatmap(cm / np.sum(cm), square=True, annot=True, fmt='.2%', cbar=False, ax=ax1)
    ax1.set_ylabel(f"{names[0]}", fontsize=14)
    cm = confusion_matrix(y_test, predictions[1])
    heatmap(cm / np.sum(cm), square=True, annot=True, fmt='.2%', cbar=False, ax=ax2)
    ax2.set_ylabel(f"{names[1]}", fontsize=14)
    cm = confusion_matrix(y_test, predictions[2])
    heatmap(cm / np.sum(cm), square=True, annot=True, fmt='.2%', cbar=False, ax=ax3)
    ax3.set_ylabel(f"{names[2]}", fontsize=14)
    cm = confusion_matrix(y_test, predictions[3])
    heatmap(cm / np.sum(cm), square=True, annot=True, fmt='.2%', cbar=False, ax=ax4)
    ax4.set_ylabel(f"{names[3]}", fontsize=14)
    plt.suptitle("BSS Demand Classification Predictions (Up or Down) - Confusion Matrices")
    plt.show()


def compareClassifiers(dataset: Dataset, random_state: int = None) -> None:
    """
    Cross validates the classifiers with a time series split then compares
    fitting times, test scores, results analyses and plots predicted
    classifications.

    :param dataset: BSS dataset, should be a Dataset
    :param random_state: Controls the random seed, should be an int
    :return: None
    """
    dataset.apply(BSS.binaryEncode, dataset.target)

    X_train, X_test, y_train, y_test = dataset.split(train_size=0.1, random_state=random_state, shuffle=False)

    models = {'GBC': ensemble.GradientBoostingClassifier(random_state=random_state),
              'RFC': ensemble.RandomForestClassifier(random_state=random_state),
              'LR': linear_model.LogisticRegression(random_state=random_state),
              'RC': linear_model.RidgeClassifier(random_state=random_state),
              'SGDC': linear_model.SGDClassifier(random_state=random_state),
              'KNC': neighbors.KNeighborsClassifier(),
              'NC': neighbors.NearestCentroid(),
              'MLPC': neural_network.MLPClassifier(random_state=random_state),
              'SVC': svm.SVC(),
              'DTC': tree.DecisionTreeClassifier(random_state=random_state)}

    results = compareModels(models, X_train, y_train)

    # removes classifiers that performed poorly
    del results['RC']
    del results['SGDC']
    del results['KNC']
    del results['NC']
    del results['MLPC']
    del results['SVC']

    plt.figure()
    _plotBox(plt, results, 'test_score', "Model Test Score Comparison")
    plt.figure()
    _plotBox(plt, results, 'fit_time', "Model Fitting Time Comparison")
    plt.show()

    predictions = []
    for name in results:
        results[name]['model'].fit(X_train, y_train)
        predictions.append((name, results[name]['model'].predict(X_test)))

    plotClassifierResultAnalysis(y_test, predictions)

    plotClassifications(y_test, *zip(*predictions))
