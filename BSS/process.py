from __future__ import annotations

import logging
import os

import pandas as pd
from pandas import DataFrame

import BSS

# Constants
local_dir = os.path.dirname(__file__)


def preProcess(df: DataFrame, location: str = '') -> DataFrame | None:
    """
    Pre-Process the DC dataset, by generalising feature names,
    normalising values and dropping irrelevant features.
    :param df: DataFrame
    :param location: str
    :return:
        - df - DataFrame | None
    """
    logging.info("Pre-Processing data")
    # TODO: Normalise values
    if df is None:
        logging.warning(f"'DataFrame' object was expected, got {type(df)}")
        return

    if location == 'DC':
        # Renaming features
        df.rename(columns={'dteday': 'datetime', 'holiday': 'is_holiday', 'weathersit': 'weather_code',
                           'windspeed': 'wind_speed'}, inplace=True)

        # Reduce dimensionality
        df = df.drop(['instant', 'casual', 'registered'], axis=1)
        df['is_weekend'] = [1 if x in [6, 0] else 0 for x in df['weekday']]
        df = df.drop(['weekday', 'workingday'], axis=1)

    elif location == 'London':
        # Renaming features
        df.rename(columns={'timestamp': 'datetime', 't1': 'temp', 't2': 'atemp'}, inplace=True)

    return df


def processData(df: DataFrame = None) -> DataFrame | None:
    """
    Processes and adapts the BSS dataset to suit time series
    by adding a datetime index and historical data.
    :param df: DataFrame
    :return:
        - df - DataFrame | None
    """
    logging.info("Processing data")
    if df is None:
        logging.warning(f"'DataFrame' object was expected, got {type(df)}")
        return

    # Adapts the dataset for time series
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.index = df['datetime']
    df = df.drop('datetime', axis=1)

    # Adding historical data
    df.loc[:, 'prev'] = df.loc[:, 'cnt'].shift()
    df.loc[:, 'diff'] = df.loc[:, 'prev'].diff()
    df.loc[:, 'prev-2'] = df.loc[:, 'prev'].shift()
    df.loc[:, 'diff-2'] = df.loc[:, 'prev-2'].diff()

    df = BSS.dataset.handleMissingData(df)

    # Changes datatypes
    for col in ['prev', 'diff', 'prev-2', 'diff-2']:
        df[col] = df[col].astype('int64')
    return df


def main(dir_=local_dir):
    config = BSS.Config(dir_)

    # Loads the BSS datasets
    dc = BSS.Dataset(config, name='Bike-Sharing-Dataset-day')
    if dc.df is None:
        logging.warning(f"DataFrame object was expected, got {type(dc.df)}")
        return
    london = BSS.Dataset(config, name='london-merged-hour')
    if london.df is None:
        logging.warning(f"DataFrame object was expected, got {type(london.df)}")
        return

    # Pre-process the datasets
    print(dc.df.head())
    dc.apply(preProcess, 'DC')
    dc.update(suffix='-pre-processed')

    london.apply(preProcess, 'London')
    london.update(suffix='-pre-processed')

    # Saves the pre-processed datasets
    dc.save()
    london.save()

    # Processes the datasets and handle missing data
    dc.apply(processData)
    dc.update(suffix='-processed')
    print(dc.df.axes)
    print(dc.df.head())
    print(dc.df.dtypes)

    london.apply(processData)
    london.update(suffix='-processed')


if __name__ == '__main__':
    main()
