from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series

import machine_learning as ml
from machine_learning import Dataset, utils

sns.set(style='darkgrid')


def preProcess(df: DataFrame, name: str) -> DataFrame:
    """
    Pre-Process the BSS dataset, by generalising feature names, correcting
    datatypes, normalising values and handling invalid instances.

    :param df: BSS dataset, should be a DataFrame
    :param name: Dataset's filename, should be a str
    :return: df - DataFrame
    """
    df = ml.handleMissingData(df)

    if 'DC' in name:
        # Renaming features
        df.rename(columns={'dteday': 'datetime', 'holiday': 'is_holiday', 'weathersit': 'weather_code',
                           'windspeed': 'wind_speed'}, inplace=True)

        if 'hour' in name:
            df['hr'] = pd.to_datetime(df['hr'], format='%H').dt.strftime('%H:%M:%S')
            df['datetime'] = df['datetime'] + ' ' + df['hr']
            df.drop('hr', axis=1, inplace=True)

        df['datetime'] = pd.to_datetime(df['datetime'])

        # Reduce dimensionality
        df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x in [6, 0] else 0)
        df.drop(['instant', 'weekday', 'workingday'], axis=1, inplace=True)

    elif 'London' in name:
        # Renaming features
        df.rename(columns={'timestamp': 'datetime', 't1': 'temp', 't2': 'atemp'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])

        df['yr'] = pd.DatetimeIndex(df['datetime']).year
        df['mnth'] = pd.DatetimeIndex(df['datetime']).month

        for col in ['weather_code', 'is_holiday', 'is_weekend', 'season']:
            df[col] = df[col].astype('int64')

        # Corrects the season series pattern
        df['season'] = df['season'].apply(lambda x: abs(4 - x) if x > 2 else x + 2)

        # Corrects the weather codes
        df['weather_code'] = df['weather_code'].apply(
            lambda x: {1: 1, 2: 2, 3: 2, 4: 2, 7: 3, 10: 4, 26: 4, 94: 4}.get(x))

        # Normalises values
        df['temp'] = df['temp'].apply(lambda x: (x - -8) / (39 - -8))
        df['atemp'] = df['atemp'].apply(lambda x: (x - -16) / (50 - -16))
        df['hum'] = df['hum'].apply(lambda x: round(min(100, max(0, x)) / 100, 4))
        df['wind_speed'] = df['wind_speed'].apply(lambda x: round(min(67, max(0, x)) / 67, 4))

    df.set_index('datetime', drop=False, inplace=True)

    logging.info(f"Pre-Processed {name} dataset")
    return df


def exploratoryDataAnalysis(df: DataFrame, dataset_name: str = '', dir_: str = '') -> None:
    """
    Performs initial investigations on data to discover patterns, to spot
    anomalies, to test hypothesis and to check assumptions with the help
    of summary statistics and graphical representations.

    :param df: BSS dataset, should be a DataFrame
    :param dataset_name: Name of dataset, should be a str
    :param dir_: Save location for figures, should be a str
    :return: None
    """
    logging.info("Exploratory Data Analysis")

    fig = plt.figure(figsize=(9.5, 7.5))
    graph = sns.heatmap(df.corr(), annot=True, square=True, cmap='Greens', fmt='.2f')
    graph.set_xticklabels(graph.get_xticklabels(), rotation=40)
    fig.suptitle(f"Corresponding Matrix - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))

    # Month/Day Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={'width_ratios': [1, 3]})
    graph = sns.barplot(x='mnth', y='cnt', data=df, ax=ax1)
    graph.set(xlabel='Month', ylabel='Demand')
    graph = sns.lineplot(x=pd.DatetimeIndex(df.index).day, y='cnt', data=df, ax=ax2)
    graph.set(xlabel='Day', ylabel=None)
    fig.suptitle(f"Demand Vs (Month and Day) - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))

    # Groups BSS hourly instances into summed days, makes it easier to
    #   plot the line graph.
    temp = Series(df['cnt'], index=df.index).resample('D').sum()

    # Plots BSS daily demand
    fig = plt.figure(figsize=(10, 5))
    plt.plot(temp.index, temp, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    fig.suptitle(f"Demand Vs Date - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))

    # Plots BSS Demand of weather codes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey='row', gridspec_kw={'width_ratios': [2, 1]})
    graph = sns.barplot(x=df['atemp'].apply(lambda x: round(x, 1)), y='cnt', data=df, ax=ax1)
    graph.set(xlabel='Feel Temperature', ylabel='Demand')
    graph = sns.barplot(x='weather_code', y='cnt', data=df, ax=ax2)
    graph.set(xlabel='Weather Codes', ylabel=None)

    # Plots BSS Demand of weekday and weekend
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 6), sharey='row', gridspec_kw={'width_ratios': [2, 1, 1]})
    graph = sns.boxplot(x='season', y='cnt', data=df, ax=ax1)
    graph.set(xlabel=None, ylabel='Demand')
    graph.set_xticklabels(['Winter', 'Spring', 'Summer', 'Fall'], rotation=30)
    graph = sns.boxplot(x='is_holiday', y='cnt', data=df, ax=ax2)
    graph.set(xlabel=None, ylabel=None)
    graph.set_xticklabels(['Workingday', 'Holiday'], rotation=30)
    graph = sns.boxplot(x='is_weekend', y='cnt', data=df, ax=ax3)
    graph.set(xlabel=None, ylabel=None)
    graph.set_xticklabels(['Weekday', 'Weekend'], rotation=30)
    fig.suptitle(f"Demand Vs (Season, isHoliday, isWeekend) - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))

    # Plots Normalised Environmental Values
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(6, 5), sharey='row')
    graph = sns.boxplot(y=df['temp'], ax=ax1)
    graph.set(title='Temp', ylabel=None)
    graph = sns.boxplot(y=df['atemp'], ax=ax2)
    graph.set(title='Feel Temp', ylabel=None)
    graph = sns.boxplot(y=df['hum'], ax=ax3)
    graph.set(title='Humidity', ylabel=None)
    graph = sns.boxplot(y=df['wind_speed'], ax=ax4)
    graph.set(title='Wind Speed', ylabel=None)
    fig.suptitle(f"Normalised Environmental Values - {dataset_name}")
    if dir_:
        plt.savefig(utils.joinPath(dir_, fig._suptitle.get_text(), ext='.png'))

    plt.show()  # displays all figures


def processData(df: DataFrame) -> DataFrame:
    """
    Processes and adapts the BSS dataset to suit time series by adding a
    datetime index. Feature engineering has also been applied by adding
    historical data. It also includes some forms of feature selection,
    certain features will be dropped to reduce dimensionality and
    multi-collinearity which were identified in the previous corresponding
    matrix.

    :param df: BSS dataset, should be a DataFrame
    :return: df - DataFrame
    """
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    # Adds historical data
    df['prev'] = df['cnt'].shift()
    df['diff'] = df['cnt'].diff()
    df['prev-2'] = df['prev'].shift()
    df['diff-2'] = df['prev'].diff()

    df = ml.handleMissingData(df)

    # Removed to generalise for similar datasets, reduce dimensionality and multi-collinearity
    df.drop(['temp', 'casual', 'registered'], axis=1, errors='ignore', inplace=True)

    # Changes datatypes
    for col in ['prev', 'diff', 'prev-2', 'diff-2']:
        df[col] = df[col].astype('int64')

    df['season'] = df['season'].astype("category")
    df['mnth'] = df['mnth'].astype("category")

    logging.info("Dataset processed")
    return df


def processDataset(dataset: Dataset, overwrite: bool = False) -> Dataset:
    """
    Pre-processes the dataset if applicable, then processes the dataset.

    :param dataset: BSS dataset, should be a Dataset
    :param overwrite: Overwrite existing file, should be a bool
    :return: dataset - Dataset
    """
    name = dataset.name
    dataset.update(name=name + '-pre-processed')

    if overwrite or not dataset.load():
        dataset.apply(preProcess, name)
        dataset.save()

    dataset.apply(processData)
    dataset.update(name=name + '-processed')

    dataset.apply(pd.get_dummies)  # apply one hot encoding to categorical features
    return dataset


def main(dir_: str) -> None:
    """
    Pre-processes and Processes the datasets.

    :param dir_: Project's path directory, should be a str
    :return: None
    """
    datasets = ['DC_day', 'DC_hour', 'London_hour']
    for name in datasets:
        config = ml.Config(dir_, name)

        results_dir = utils.makePath(dir_, config.results_folder, 'process')

        # Loads the BSS dataset
        dataset = ml.Dataset(config.dataset)
        if not dataset.load():
            raise Exception("Failed to load dataset")

        print(dataset.df.shape)
        print(dataset.df.head())
        dataset.apply(preProcess, name)
        dataset.update(name=(name + '-pre-processed'))
        dataset.save()

        dataset.df.drop('datetime', axis=1, inplace=True)

        exploratoryDataAnalysis(dataset.df, dataset_name=dataset.name, dir_=results_dir)

        dataset.apply(processData)
        dataset.update(name=(name + '-processed'))
        print(dataset.df.axes)
        print(dataset.df.head())
        fig = plt.figure(figsize=(9.5, 7.5))
        graph = sns.heatmap(dataset.df.corr(), annot=True, square=True, cmap='Greens', fmt='.2f')
        graph.set_xticklabels(graph.get_xticklabels(), rotation=40)
        fig.suptitle(f"Corresponding Matrix - {dataset.name}")
        plt.savefig(utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))
        plt.show()

        dataset.apply(pd.get_dummies)  # apply one hot encoding to categorical features
        print(dataset.df.dtypes)
        fig = plt.figure(figsize=(14, 10))
        graph = sns.heatmap(dataset.df.corr(), annot=True, cmap='Greens', fmt='.2f', cbar=False)
        graph.set_xticklabels(graph.get_xticklabels(), rotation=40)
        fig.suptitle(f"Encoded Corresponding Matrix - {dataset.name}")
        plt.savefig(utils.joinPath(results_dir, fig._suptitle.get_text(), ext='.png'))
        plt.show()
