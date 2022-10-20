from __future__ import annotations

import logging

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn import metrics

import machine_learning as ml


def binaryEncode(df: DataFrame, target: str) -> DataFrame:
    """
    Encodes the target into a binary category where,
    0 represents a decrease from previous instance
    1 represents an increase from previous instance.

    :param df: The dataset, should be a DataFrame
    :param target: The dataset's target value, should be a str
    :return: df - DataFrame
    """
    logging.info("Encoding target variables")
    if df[target].dtype not in ['float64', 'int64']:
        raise NotImplementedError(f"The target variables type '{df[target].dtype}' are not supported")

    # binary encodes the feature
    df[target] = [int(df[target][max(0, i - 1)] < df[target][min(len(df[target]) - 1, i)])
                  for i in range(len(df[target]))]

    # checks for class imbalance
    if len(df[target] == 0) != len(df[target] == 1):
        logging.warning("Class imbalance is present")

    df[target] = df[target].astype("category")
    return df


def plotPrediction(y_test: Series, y_pred: tuple | dict | list, dataset_name: str = '', results_dir: str = '') -> None:
    """
    Plot the predictions in a confusion matrix format.

    :param y_test: Testing dependent variables, should be a Series
    :param y_pred: Classifier predictions, should be a tuple[str, ndarray] | dict[str: ndarray
     | list[tuple[str, ndarray]
    :param dataset_name: Name of dataset, should be a str
    :param results_dir: Save location for figures, should be a str
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

    if len(y_preds) == 3:
        names, y_preds = zip(*y_preds)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row')
        cm = metrics.confusion_matrix(y_test, y_preds[0])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax1)
        graph.set(xlabel=names[0])
        cm = metrics.confusion_matrix(y_test, y_preds[1])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax2)
        graph.set(xlabel=names[1])
        cm = metrics.confusion_matrix(y_test, y_preds[2])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax3)
        graph.set(xlabel=names[2])
        fig.suptitle(f"Classifier Prediction - {dataset_name}")
        if results_dir:
            plt.savefig(ml.utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))
    elif len(y_preds) == 4:
        names, y_preds = zip(*y_preds)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        cm = metrics.confusion_matrix(y_test, y_preds[0])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax1)
        graph.set(xlabel=names[0])
        cm = metrics.confusion_matrix(y_test, y_preds[1])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax2)
        graph.set(xlabel=names[1])
        cm = metrics.confusion_matrix(y_test, y_preds[2])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax3)
        graph.set(xlabel=names[2])
        cm = metrics.confusion_matrix(y_test, y_preds[3])
        graph = sns.heatmap(cm / np.sum(cm), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False, ax=ax4)
        graph.set(xlabel=names[3])
        fig.suptitle(f"Classifier Prediction - {dataset_name}")
        if results_dir:
            plt.savefig(ml.utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))
    else:
        for name, y_pred in y_preds:
            fig, ax = plt.subplots()
            cm = metrics.confusion_matrix(y_test, y_pred)
            graph = sns.heatmap((cm / np.sum(cm)), annot=True, square=True, cmap='Greens', fmt='.2%', cbar=False)
            graph.set(xlabel=name)
            fig.suptitle(f"{name} Classifier Prediction - {dataset_name}")
            if results_dir:
                plt.savefig(ml.utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))
    plt.show()


def resultAnalysis(y_test: Series, y_pred: tuple | dict | list, plot: bool = True, display: bool = True,
                   dataset_name: str = '', results_dir: str = '') -> dict:
    """
    Calculate the result analysis with options to display and plot.

    :param y_test: Testing dependent variables, should be a Series
    :param y_pred: Model predictions, should be a tuple[str, ndarray] | dict[str: ndarray
     | list[tuple[str, ndarray]
    :param plot: Whether to plot the results, should be a bool
    :param display: Whether to show the results, should be a bool
    :param dataset_name: Name of dataset, should be a str
    :param results_dir: Save location for figures, should be a str
    :return: results - dict[str: list[str | float]]
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
        results['accuracy'].append(metrics.accuracy_score(y_test, y_pred))
        results['precision'].append(metrics.precision_score(y_test, y_pred))
        results['recall'].append(metrics.recall_score(y_test, y_pred))
        results['f1'].append(metrics.f1_score(y_test, y_pred))

        if display:
            print("\nModel:", name)
            print("Accuracy: %.4f" % results['accuracy'][-1])
            print("Precision: %.4f" % results['precision'][-1])
            print("Recall: %.4f" % results['recall'][-1])
            print("F1: %.4f" % results['f1'][-1])

    if not plot:
        return results

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), sharex='col')
    ml.utils._plotBar(ax1, results['names'], results['accuracy'], 'Accuracy')
    ml.utils._plotBar(ax2, results['names'], results['precision'], 'Precision')
    ml.utils._plotBar(ax3, results['names'], results['recall'], 'Recall')
    ml.utils._plotBar(ax4, results['names'], results['f1'], 'F1')
    fig.suptitle(f"Result Analysis - {dataset_name}")
    if results_dir:
        plt.savefig(ml.utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))
    plt.show()
    return results
