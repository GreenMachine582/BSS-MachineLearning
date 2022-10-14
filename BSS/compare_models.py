from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn import ensemble, neighbors, neural_network, svm, tree, linear_model
from sklearn.model_selection import TimeSeriesSplit, cross_validate

import machine_learning as ml
from machine_learning import Dataset, Config


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
        plt.savefig(ml.utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
    plt.show()
    return results


def compareEstimators(dataset: Dataset, config: Config) -> None:
    """
    Cross validates the estimators with a time series split then compares
    fitting times, test scores, results analyses and plots predicted
    estimations.

    :param dataset: BSS dataset, should be a Dataset
    :param config: BSS configuration, should be a Config
    :return: None
    """
    estimators = {'GBR': ensemble.GradientBoostingRegressor(random_state=config.random_state),
                  'RFR': ensemble.RandomForestRegressor(random_state=config.random_state),
                  'KNR': neighbors.KNeighborsRegressor(),
                  'MLPR': neural_network.MLPRegressor(max_iter=2400, random_state=config.random_state),
                  'SVR': svm.SVR(),
                  'DTR': tree.DecisionTreeRegressor(random_state=config.random_state),
                  'ETR': tree.ExtraTreeRegressor(random_state=config.random_state)}

    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    results_dir = ml.utils.makePath(config.dir_, config.results_folder, 'compare_estimators')

    results = compareModels(estimators, X_train, y_train, dataset_name=dataset.name, dir_=results_dir)

    # removes estimators that performed poorly
    del results['KNR']
    del results['SVR']
    del results['DTR']
    del results['ETR']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
    ax1.boxplot([results[name]['test_score'] for name in results], labels=[name for name in results])
    ax1.set(ylabel="Testing Score")
    ax2.boxplot([results[name]['fit_time'] for name in results], labels=[name for name in results])
    ax2.set(ylabel="Fitting Time")
    fig.suptitle(f"Model Comparison (Closeup) - {dataset.name}")
    if results_dir:
        plt.savefig(ml.utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))
    plt.show()
    logging.info("Fitting and predicting")
    y_preds = []
    for name in results:
        results[name]['model'].fit(X_train, y_train)
        y_pred = results[name]['model'].predict(X_test)
        y_preds.append((name, np.clip(y_pred, 0, None)))

    ml.estimator.resultAnalysis(y_test, y_preds, dataset_name=dataset.name, dir_=results_dir)
    y_preds = [(name, Series(y_pred, index=y_test.index).resample('D').sum()) for name, y_pred in y_preds]
    ml.estimator.plotPrediction(y_train.resample('D').sum(), y_test.resample('D').sum(), y_preds, target=dataset.target,
                                dataset_name=dataset.name, dir_=results_dir)


def compareClassifiers(dataset: Dataset, config: Config) -> None:
    """
    Cross validates the classifiers with a time series split then compares
    fitting times, test scores, results analyses and plots predicted
    classifications.

    :param dataset: BSS dataset, should be a Dataset
    :param config: BSS configuration, should be a Config
    :return: None
    """
    dataset.df.drop('diff', axis=1, inplace=True)  # same feature variables as binary encoded target
    dataset.apply(ml.binaryEncode, dataset.target)

    models = {'GBC': ensemble.GradientBoostingClassifier(random_state=config.random_state),
              'RFC': ensemble.RandomForestClassifier(random_state=config.random_state),
              'LR': linear_model.LogisticRegression(max_iter=1000, random_state=config.random_state),
              'RC': linear_model.RidgeClassifier(random_state=config.random_state),
              'SGDC': linear_model.SGDClassifier(random_state=config.random_state),
              'KNC': neighbors.KNeighborsClassifier(),
              'NC': neighbors.NearestCentroid(),
              'MLPC': neural_network.MLPClassifier(max_iter=300, random_state=config.random_state),
              'SVC': svm.SVC(),
              'DTC': tree.DecisionTreeClassifier(random_state=config.random_state),
              'ETC': tree.ExtraTreeClassifier()}

    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    results_dir = ml.utils.makePath(config.dir_, config.results_folder, 'compare_classifiers')

    results = compareModels(models, X_train, y_train, dataset_name=dataset.name, dir_=results_dir)

    # removes classifiers that performed poorly
    del results['LR']
    del results['RC']
    del results['SGDC']
    del results['NC']
    del results['SVC']
    del results['DTC']
    del results['ETC']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
    ax1.boxplot([results[name]['test_score'] for name in results], labels=[name for name in results])
    ax1.set(ylabel="Testing Score")
    ax2.boxplot([results[name]['fit_time'] for name in results], labels=[name for name in results])
    ax2.set(ylabel="Fitting Time")
    fig.suptitle(f"Model Comparison (Closeup) - {dataset.name}")
    if results_dir:
        plt.savefig(ml.utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))
    plt.show()

    y_preds = []
    for name in results:
        results[name]['model'].fit(X_train, y_train)
        y_preds.append((name, results[name]['model'].predict(X_test)))

    ml.classifier.resultAnalysis(y_test, y_preds, dataset_name=dataset.name, dir_=results_dir)
    ml.classifier.plotPrediction(y_test, y_preds, dataset_name=dataset.name, dir_=results_dir)
