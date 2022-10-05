from __future__ import annotations

import logging
import os

from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame, Series
from seaborn import heatmap
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Constants
local_dir = os.path.dirname(__file__)


def convertToCategorical(df: DataFrame) -> DataFrame:
    """
    Encode the BSS cnt into a binary category where,
    0 represents a fall in price from previous instance
    1 represents a rise in price from previous instance.

    :param df: The BTC dataset, should be a DataFrame
    :return: df - DataFrame
    """
    df['cnt'] = [0 if df['diff'][i] < 0 else 1 for i in range(len(df['cnt']))]

    # Checks for class imbalance
    if len(df['cnt'] == 0) != len(df['cnt'] == 1):
        logging.warning("Class imbalance is present")

    df['cnt'] = df['cnt'].astype("category")
    return df


def plotClassifications(y_test: Series, name: str, y_preds: ndarray) -> None:
    """
    Plot the predictions in a confusion matrix format.

    :param y_test: Testing dependent variables, should be a Series
    :param name: Classifier's name, should be a str
    :param y_preds: Classifier predictions, should be a ndarray
    :return: None
    """
    plt.figure()
    df_cm = DataFrame(confusion_matrix(y_test, y_preds))
    heatmap(df_cm, square=True, annot=True, fmt='d', cbar=False)
    plt.suptitle(f"BSS Demand Classification Predictions - {name} - Confusion Matrix")
    plt.show()


def resultAnalysis(y_test: Series, y_pred: ndarray, show: bool = True) -> dict:
    """
    Calculate and display the result analysis for classifiers.

    :param y_test: Testing dependent variables, should be a Series
    :param y_pred: Model predictions, should be a ndarray
    :param show: Whether to show the results, should be a bool
    :return: results - dict[str: float]
    """
    logging.info("Analysing results")

    results = {'accuracy': accuracy_score(y_test, y_pred),
               'precision': precision_score(y_test, y_pred),
               'recall': recall_score(y_test, y_pred),
               'f1': f1_score(y_test, y_pred)}

    if show:
        print('Accuracy: %.4f' % results['accuracy'])
        print('Precision: %.4f' % results['precision'])
        print('Recall: %.4f' % results['recall'])
        print('F1: %.4f' % results['f1'])
    return results
