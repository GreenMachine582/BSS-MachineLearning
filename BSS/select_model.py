from __future__ import annotations

import logging
import os

from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import ensemble, linear_model, neighbors, neural_network, svm, tree
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import BSS
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def convertToCategorical(df: DataFrame) -> DataFrame:
    """
    Converts the BSS demand (cnt) into a binary category where,
    0 represents a fall in demand from previous instance
    1 represents a rise in demand from previous instance.
    :param df: The BSS dataset, should be a DataFrame
    :return: df - DataFrame
    """
    df['cnt'] = [0 if df['diff'][i] < 0 else 1 for i in range(len(df['cnt']))]

    # Checks for class imbalance
    if len(df['cnt'] == 0) != len(df['cnt'] == 1):
        logging.warning("Class imbalance is present")

    df['cnt'] = df['cnt'].astype("category")
    return df


def getEstimators(random_state: int = None) -> list:
    models = [('EGBR', ensemble.GradientBoostingRegressor(random_state=random_state)),
              ('ERFR', ensemble.RandomForestRegressor(random_state=random_state)),
              ('NKNR', neighbors.KNeighborsRegressor()),
              ('NNMLPR', neural_network.MLPRegressor(max_iter=1200, random_state=random_state)),
              ('SSVR', svm.SVR()),
              ('TDTR', tree.DecisionTreeRegressor(random_state=random_state)),
              ('TETR', tree.ExtraTreeRegressor(random_state=random_state))]
    return models


def getClassifiers(random_state: int = None) -> list:
    models = [('EGBC', ensemble.GradientBoostingClassifier(random_state=random_state)),
              ('ERFC', ensemble.RandomForestClassifier(random_state=random_state)),
              ('LMLR', linear_model.LogisticRegression(random_state=random_state)),
              ('LMRC', linear_model.RidgeClassifier(random_state=random_state)),
              ('LMSGDC', linear_model.SGDClassifier(random_state=random_state)),
              ('NKNC', neighbors.KNeighborsClassifier()),
              ('NNC', neighbors.NearestCentroid()),
              ('NNMLPC', neural_network.MLPClassifier(max_iter=1200, random_state=random_state)),
              ('SSVC', svm.SVC()),
              ('TDTC', tree.DecisionTreeClassifier(random_state=random_state))]
    return models


def compareModels(X: DataFrame, y: DataFrame, models: list, cv: int | object = 10) -> dict:
    """
    Trains and cross validates a basic model with default params.

    :param X: Independent features, should be a DataFrame
    :param y: Dependent variables, should be a DataFrame
    :param models:
    :param cv:
    :return: results - dict[str: cv_results]
    """
    # TODO: Documentation
    logging.info("Comparing models")
    results = {}
    for name, model in models:
        cv_results = cross_val_score(model, X, y, cv=cv)
        results[name] = cv_results
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    return results


def main(dir_: str = local_dir) -> None:
    """

    :param dir_: project's path directory, should be a str
    :return: None
    """
    # TODO: Documentation and error handling
    name = 'Bike-Sharing-Dataset-hour'
    config = ml.Config(dir_, name)

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        return

    dataset = BSS.processDataset(dataset)

    logging.info('Select Group of Models')
    while True:
        print("""
            0 - Back
            1 - Estimators
            2 - Classifiers
            """)
        choice = input("Which option number: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        if choice is not None:
            if choice == 0:
                return
            elif choice == 1:
                models = getEstimators(config.random_state)
                break
            elif choice == 2:
                dataset.apply(convertToCategorical)
                models = getClassifiers(config.random_state)
                break
            else:
                print("\nPlease enter a valid choice!")

    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    results = compareModels(X_train, y_train, models, cv=TimeSeriesSplit(n_splits=10))

    plt.figure()
    plt.boxplot(results.values(), labels=results.keys())
    plt.title('Algorithm Comparison')
    plt.show()

    # Remove models that performed poorly
    if choice == 1:
        del results['SSVR']
        del results['TDTR']
        del results['TETR']

        plt.figure()
        plt.boxplot(results.values(), labels=results.keys())
        plt.title('Algorithm Comparison')
        plt.show()
    else:
        del results['LMRC']
        del results['NKNC']
        del results['NNC']
        del results['SSVC']

        plt.figure()
        plt.boxplot(results.values(), labels=results.keys())
        plt.title('Algorithm Comparison')
        plt.show()

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
