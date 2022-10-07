from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series

import machine_learning as ml
from machine_learning import Dataset

sns.set(style='darkgrid')


def preProcess(df: DataFrame, name: str) -> DataFrame:
    """
    Pre-Process the BSS dataset, by generalising feature names, correcting
    datatypes, normalising values and handling invalid instances.

    :param df: BSS dataset, should be a DataFrame
    :param name: dataset's filename, should be a str
    :return: df - DataFrame
    """
    df = ml.handleMissingData(df)

    if 'Bike-Sharing-Dataset' in name:
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

    elif 'london_merged' in name:
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


def exploratoryDataAnalysis(df: DataFrame) -> None:
    """
    Performs initial investigations on data to discover patterns, to spot
    anomalies, to test hypothesis and to check assumptions with the help
    of summary statistics and graphical representations.

    :param df: BSS dataset, should be a DataFrame
    :return: None
    """
    logging.info("Exploratory Data Analysis")

    plt.figure()
    sns.heatmap(df.corr(), annot=True)
    plt.title('Pre-Processed Corresponding Matrix')

    # Month/Day Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6), gridspec_kw={'width_ratios': [1, 3]})
    sns.barplot(x='mnth', y='cnt', data=df, ax=ax1)
    ax1.set_xlabel('Month', fontsize=14)
    ax1.set_ylabel('Cnt', fontsize=14)
    sns.lineplot(pd.DatetimeIndex(df.index).day, y='cnt', data=df, ax=ax2)
    ax2.set_xlabel('Day', fontsize=14)
    ax2.set_ylabel('Cnt', fontsize=14)
    plt.suptitle("BSS Demand")

    # Line Plot
    plt.figure()
    # Groups BSS hourly instances into summed days, makes it easier to
    #   plot the line graph.
    temp = Series(df['cnt'], index=df.index).resample('D').sum()
    plt.plot(temp.index, temp, color='blue')
    plt.title('BSS Demand Vs Date')
    plt.xlabel('Datetime')
    plt.ylabel('Cnt')
    plt.show()

    # Bar Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6), sharey='row', gridspec_kw={'width_ratios': [2, 1]})
    sns.barplot(df['atemp'].apply(lambda x: round(x, 1)), y='cnt', data=df, ax=ax1)
    ax1.set_xlabel('Feel Temperature', fontsize=14)
    ax1.set_ylabel('Cnt', fontsize=14)
    sns.barplot(x='weather_code', y='cnt', data=df, ax=ax2)
    ax2.set_xlabel('Weather Codes', fontsize=14)
    plt.suptitle("BSS Demand")

    # Box plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 6), sharey='row', gridspec_kw={'width_ratios': [2, 1, 1]})
    sns.boxplot(x='season', y='cnt', data=df, ax=ax1)
    ax1.set_xlabel('Season', fontsize=14)
    ax1.set_ylabel('Cnt', fontsize=14)
    sns.boxplot(x='is_holiday', y='cnt', data=df, ax=ax2)
    ax2.set_xlabel('Workingday Vs Holiday', fontsize=14)
    sns.boxplot(x='is_weekend', y='cnt', data=df, ax=ax3)
    ax3.set_xlabel('Weekday Vs Weekend', fontsize=14)
    plt.suptitle("BSS Demand")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(6, 4), sharey='row')
    sns.boxplot(y=df['temp'], ax=ax1)
    ax1.set_ylabel('Temp', fontsize=14)
    sns.boxplot(y=df['atemp'], ax=ax2)
    ax2.set_ylabel('Feel Temp', fontsize=14)
    sns.boxplot(y=df['hum'], ax=ax3)
    ax3.set_ylabel('Humidity', fontsize=14)
    sns.boxplot(y=df['wind_speed'], ax=ax4)
    ax4.set_ylabel('Wind Speed', fontsize=14)
    plt.suptitle("Normalised Environmental Values")

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
    df['prev'] = df['Close'].shift()
    df['diff'] = df['Close'].diff()
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

    :param dataset: the dataset, should be a Dataset
    :param overwrite: overwrite existing file, should be a bool
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

    :param dir_: project's path directory, should be a str
    :return: None
    """
    datasets = ['Bike-Sharing-Dataset-day', 'Bike-Sharing-Dataset-hour', 'london_merged-hour']
    for name in datasets:
        config = ml.Config(dir_, name)

        # Loads the BSS dataset
        dataset = ml.Dataset(config.dataset)
        if not dataset.load():
            return

        print(dataset.df.shape)
        print(dataset.df.head())
        dataset.apply(preProcess, name)
        dataset.update(name=(name + '-pre-processed'))
        dataset.save()

        dataset.df.drop('datetime', axis=1, inplace=True)

        exploratoryDataAnalysis(dataset.df)

        dataset.apply(processData)
        dataset.update(name=(name + '-processed'))
        print(dataset.df.axes)
        print(dataset.df.head())
        plt.figure()
        sns.heatmap(dataset.df.corr(), annot=True)
        plt.title('Processed Corresponding Matrix')
        plt.show()

        dataset.apply(pd.get_dummies)  # apply one hot encoding to categorical features
        print(dataset.df.dtypes)
        plt.figure(figsize=(14, 10))
        sns.heatmap(dataset.df.corr(), annot=True)
        plt.title('Processed and Encoded Corresponding Matrix')
        plt.show()
