from __future__ import annotations

import logging

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from machine_learning import utils


def binaryEncode(df: DataFrame, target: str) -> DataFrame:
    """
    Encodes the target into a binary category where,
    0 represents a decrease from previous instance
    1 represents an increase from previous instance.

    :param df: The dataset, should be a DataFrame
    :param target: The dataset's target value, should be a str
    :return: df - DataFrame
    """
    if df[target].dtype not in ['float64', 'int64']:
        logging.warning("The target values are not compatible")
        return df

    df[target] = [int(df[target][max(0, i - 1)] < df[target][min(len(df[target]) - 1, i)])
                  for i in range(len(df[target]))]

    # Checks for class imbalance
    if len(df[target] == 0) != len(df[target] == 1):
        logging.warning("Class imbalance is present")

    df[target] = df[target].astype("category")
    return df


def plotPredictions(y_test: Series, y_pred: tuple | dict | list, dataset_name: str = '',
                    dir_: str = '') -> None:
    """
    Plot the predictions in a confusion matrix format.

    :param y_test: Testing dependent variables, should be a Series
    :param y_pred: Classifier predictions, should be a tuple[str, ndarray] | dict[str: ndarray
     | list[tuple[str, ndarray]
    :param dataset_name: Name of dataset, should be a str
    :param dir_: Save location for figures, should be a str
    :return: None
    """
    logging.info("Plotting predictions")

    if isinstance(y_pred, tuple):
        y_preds = [y_pred]
    elif isinstance(y_pred, dict):
        y_preds = [(x, y_pred[x]) for x in y_pred]
    elif isinstance(y_pred, list):
        y_preds = y_pred
    else:
        raise TypeError(f"'y_pred': Expected type 'tuple | dict | list', got {type(y_pred).__name__} instead")

    if len(y_preds) == 4:
        names, y_preds = zip(*y_preds)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        cm = confusion_matrix(y_test, y_preds[0])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax1)
        graph.set(xlabel=names[0])
        cm = confusion_matrix(y_test, y_preds[1])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax2)
        graph.set(xlabel=names[1])
        cm = confusion_matrix(y_test, y_preds[2])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax3)
        graph.set(xlabel=names[2])
        cm = confusion_matrix(y_test, y_preds[3])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax4)
        graph.set(xlabel=names[3])
        fig.suptitle(f"BSS Predicted Demand (Up or Down) - {dataset_name}")
        if dir_:
            plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
    else:
        for name, y_pred in y_preds:
            fig, ax = plt.subplots()
            cm = DataFrame(confusion_matrix(y_test, y_pred))
            graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False)
            graph.set(xlabel=name)
            fig.suptitle(f"BSS {name} Predicted Demand (Up or Down) - {dataset_name} - Confusion Matrix")
            if dir_:
                plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
    plt.show()


def plotResultAnalysis(y_test: Series, y_pred: tuple | dict | list, show: bool = True, dataset_name: str = '',
                       dir_: str = '') -> None:
    """
    Calculate and display the result analysis for classifiers.

    :param y_test: Testing dependent variables, should be a Series
    :param y_pred: Model predictions, should be a tuple[str, ndarray] | dict[str: ndarray
     | list[tuple[str, ndarray]
    :param show: Whether to show the results, should be a bool
    :param dataset_name: Name of dataset, should be a str
    :param dir_: Save location for figures, should be a str
    :return: None
    """
    logging.info("Analysing results")

    if isinstance(y_pred, tuple):
        y_preds = [y_pred]
    elif isinstance(y_pred, dict):
        y_preds = [(x, y_pred[x]) for x in y_pred]
    elif isinstance(y_pred, list):
        y_preds = y_pred
    else:
        raise TypeError(f"'y_pred': Expected type 'tuple | dict | list', got {type(y_pred).__name__} instead")

    results = {'names': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for name, y_pred in y_preds:
        results['names'].append(name)
        results['accuracy'].append(accuracy_score(y_test, y_pred))
        results['precision'].append(precision_score(y_test, y_pred))
        results['recall'].append(recall_score(y_test, y_pred))
        results['f1'].append(f1_score(y_test, y_pred))

        if show:
            print("\nModel:", name)
            print("Accuracy: %.4f" % results['accuracy'][-1])
            print("Precision: %.4f" % results['precision'][-1])
            print("Recall: %.4f" % results['recall'][-1])
            print("F1: %.4f" % results['f1'][-1])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), sharex='col')
    utils._plotBar(ax1, results['names'], results['accuracy'], 'Accuracy')
    utils._plotBar(ax2, results['names'], results['precision'], 'Precision')
    utils._plotBar(ax3, results['names'], results['recall'], 'Recall')
    utils._plotBar(ax4, results['names'], results['f1'], 'F1')
    fig.suptitle(f"Result Analysis - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))
    plt.show()
