from __future__ import annotations

import logging
import os

import pandas as pd
from pandas import DataFrame

from . import Config, Dataset, handleMissingData

# Constants
local_dir = os.path.dirname(__file__)


def preProcess(df: DataFrame, name: str) -> DataFrame:
    """
    Pre-Process the DC dataset, by generalising feature names,
    normalising values and handling invalid instances.

    :param df: the dataset, should be a DataFrame
    :param name: dataset's filename, should be a str
    :return: df - DataFrame
    """
    df = handleMissingData(df)

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
        df.drop(['instant', 'casual', 'registered'], axis=1, inplace=True)
        df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x in [6, 0] else 0)
        df.drop(['weekday', 'workingday'], axis=1, inplace=True)

    elif 'london_merged' in name:
        # Renaming features
        df.rename(columns={'timestamp': 'datetime', 't1': 'temp', 't2': 'atemp'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])

        df['yr'] = pd.DatetimeIndex(df['datetime']).year
        df['mnth'] = pd.DatetimeIndex(df['datetime']).month

        for col in ['weather_code', 'is_holiday', 'is_weekend', 'season']:
            df[col] = df[col].astype('int64')

        df['season'] = df['season'].apply(lambda x: abs(4 - x) if x > 2 else x + 2)

        df['weather_code'] = df['weather_code'].apply(
            lambda x: {1: 1, 2: 2, 3: 2, 4: 2, 7: 3, 10: 4, 26: 4, 94: 4}.get(x))

        df['temp'] = df['temp'].apply(lambda x: (x - -8) / (39 - -8))
        df['atemp'] = df['atemp'].apply(lambda x: (x - -16) / (50 - -16))

        df['hum'] = df['hum'].apply(lambda x: round(min(100, max(0, x)) / 100, 4))
        df['wind_speed'] = df['wind_speed'].apply(lambda x: round(min(67, max(0, x)) / 67, 4))

    logging.info(f"Pre-Processed {name} dataset")
    return df


def processData(df: DataFrame) -> DataFrame:
    """
    Processes and adapts the BSS dataset to suit time series by adding
    a datetime index and historical data.

    :param df: the dataset, should be a DataFrame
    :return: df - DataFrame
    """
    # Adapts the dataset for time series
    df.index = df['datetime']
    df.drop(['datetime'], axis=1, inplace=True)

    # Adding historical data
    df.loc[:, 'prev'] = df.loc[:, 'cnt'].shift()
    df.loc[:, 'diff'] = df.loc[:, 'prev'].diff()
    df.loc[:, 'prev-2'] = df.loc[:, 'prev'].shift()
    df.loc[:, 'diff-2'] = df.loc[:, 'prev-2'].diff()

    df = handleMissingData(df)

    # Changes datatypes
    for col in ['prev', 'diff', 'prev-2', 'diff-2']:
        df[col] = df[col].astype('int64')

    df['season'] = df['season'].astype("category")
    df['mnth'] = df['mnth'].astype("category")

    df = pd.get_dummies(df)  # apply one hot encoding to categorical features

    logging.info("Dataset processed")
    return df


def main(dir_: str = local_dir) -> None:
    """
    Pre-processes and Processes the datasets before feature
    engineering/selection and Machine Learning are applied.

    :param dir_: project's path directory, should be a str
    :return:
    """
    datasets = ['Bike-Sharing-Dataset-day', 'Bike-Sharing-Dataset-hour', 'london_merged-hour']
    for name in datasets:
        config = Config(dir_, name)
        if not config.load():
            config.save()

        # Loads the BSS dataset
        dataset = Dataset(config.dataset)
        if not dataset.load():
            return

        # Pre-process the dataset
        print(dataset.df.head())
        dataset.apply(preProcess, dataset.name)
        dataset.update(name=name + '-pre-processed')
        dataset.save()

        # Processes the dataset and handle missing data
        dataset.apply(processData)
        print(dataset.df.axes)
        print(dataset.df.head())
        print(dataset.df.dtypes)


if __name__ == '__main__':
    main()
