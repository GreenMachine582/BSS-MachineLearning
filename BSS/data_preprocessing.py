#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data_preprocessing.py: Take datasets given in /examples/datasets as input data and generate a processed dataset
As data pre-processing requires a good knowledge of the target database,
the operations for pre-processing will vary greatly from database to database,
so this document is only applicable to the data given in /examples/datasets
"""

__author__ = "Yuki"
__date__ = "08/09/2022"

import logging
import os

from pandas import DataFrame

import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def dataConsolidation(df_dc, df_london):
    """
    Input: data frame of bike_dc and bike_london
    Output: data frame of merged data

    CAN ONLY WORK WITH bike_dc AND bike_london. Take bike_dc and bike_london as and merge them together
    Merged dataframe should have the following attributes:
    new    | dc     |   london  |
    season | season | season    |
    month  | mnth   | timestamp |
    day    | dteday | timestamp |
    hour   | hr     | timestamp |
    isholiday | holiday | isholiday |
    isweekend | weekday | isweekend |
    isworkingday | workingday | isholiday+isweekend |
    hum | hum | hum |
    t1 | temp | t1 |
    t2 | atemp | t2 |
    windspeed | windspeed | windspeed |
    weather | weathersit | weathercode |
    city | dc | london |
    cnt | cnt | cnt |
    rate | cnt | cnt |
    """
    logging.info('Consolidating datasets')
    # list of attributes
    alist = ["rate", "cnt", "city", "weather", "wind_speed", "t2", "t1", "hum", "is_workingday", "is_weekend",
             "is_holiday", "hour", "day", "month", "season"]
    alen = len(alist)

    # population in each city in each year
    pdc = 600000
    pld = 8750000

    # new dictionary for the new dataframe
    d = {}
    for i in range(0, alen):
        d[alist[alen - 1 - i]] = []

    # merge season
    season_dc = df_dc["season"].values.tolist()
    season_ld = df_london["season"].values.tolist()
    for i in range(0, len(season_dc)):
        if season_dc[i] == 1:
            season_dc[i] = str(3)
        else:
            season_dc[i] = str(season_dc[i] + 2 - 4)
    season = season_dc + [str(y) for y in [int(x) for x in season_ld]]
    d["season"] = season

    # merge month, day and hour
    dc_month = df_dc["mnth"].values.tolist()
    dc_day = df_dc["dteday"].values.tolist()
    for i in range(0, len(dc_day)):
        str_list = dc_day[i].split("-")
        dc_day[i] = int(str_list[2])
    dc_hr = df_dc["hr"].values.tolist()

    ld_time = df_london["timestamp"].values.tolist()
    ld_month = []
    ld_day = []
    ld_hr = []
    for i in range(0, len(ld_time)):
        m = ld_time[i].split("-")
        ld_month.append(int(m[1]))
        da = m[2].split(" ")
        ld_day.append(int(da[0]))
        hr = da[1].split(":")
        ld_hr.append(int(hr[0]))
    month = dc_month + ld_month
    day = dc_day + ld_day
    hour = dc_hr + ld_hr
    d["month"] = month
    d["day"] = day
    d["hour"] = hour

    # merge is_holiday
    d["is_holiday"] = [str(x) for x in [int(x) for x in (df_dc["holiday"].values.tolist()
                                                         + df_london["is_holiday"].values.tolist())]]

    # merge is_weekend
    dc_week = df_dc["weekday"].values.tolist()
    dc_is_weekend = []
    for i in range(0, len(dc_week)):
        if dc_week[i] == 0 or dc_week[i] == 6:
            dc_is_weekend.append(str(1))
        else:
            dc_is_weekend.append(str(0))
    ld_is_weekend = [str(x) for x in [int(x) for x in df_london["is_weekend"].values.tolist()]]
    d["is_weekend"] = dc_is_weekend + ld_is_weekend

    # merge is_workingday
    dc_is_workingday = [str(x) for x in df_dc["workingday"].values.tolist()]
    ld_is_workingday = []
    ldw = df_london["is_weekend"].values.tolist()
    ldh = df_london["is_holiday"].values.tolist()
    for i in range(0, len(ld_time)):
        if (ldw[i] + ldh[i]) == 0:
            ld_is_workingday.append(str(1))
        else:
            ld_is_workingday.append(str(0))
    d["is_workingday"] = dc_is_workingday + ld_is_workingday

    # merge hum
    d["hum"] = df_dc["hum"].values.tolist() + [x / 100 for x in df_london["hum"].values.tolist()]

    # merge t2
    dc_t2 = [x * 47 - 8 for x in df_dc["atemp"].values.tolist()]
    ld_t2 = df_london["t2"].values.tolist()
    d["t2"] = dc_t2 + ld_t2

    # merge t1
    dc_t1 = [x * 66 - 16 for x in df_dc["temp"].values.tolist()]
    ld_t1 = df_london["t1"].values.tolist()
    d["t1"] = dc_t1 + ld_t1

    # merge wind_speed
    dc_wind_speed = [x * 67 for x in df_dc["windspeed"].values.tolist()]
    ld_wind_speed = df_london["wind_speed"].values.tolist()
    d["wind_speed"] = dc_wind_speed + ld_wind_speed

    # merge weather
    # london: 1,2 = dc: 1
    # london: 3,4 = dc: 2
    # london: 7 = dc: 3
    # london: 10, 26, 94 = dc: 4
    dc_weather = [str(x) for x in df_dc["weathersit"].values.tolist()]
    ld_weather = df_london["weather_code"].values.tolist()
    for i in range(0, len(ld_weather)):
        v = ld_weather[i]
        if v == 1 or v == 2:
            ld_weather[i] = str(1)
        if v == 3 or v == 4:
            ld_weather[i] = str(2)
        if v == 7:
            ld_weather[i] = str(3)
        if v == 10 or v == 26 or v == 94:
            ld_weather[i] = str(4)
    d["weather"] = dc_weather + ld_weather

    # merge city
    ct_dc = ["DC"] * len(dc_day)
    ct_ld = ["London"] * len(ld_time)
    d["city"] = ct_dc + ct_ld

    # merge cnt
    d["cnt"] = df_dc["cnt"].values.tolist() + df_london["cnt"].values.tolist()

    # merge rate
    d["rate"] = [x / 60 for x in df_dc["cnt"].values.tolist()] + [x / 875 for x in df_london["cnt"].values.tolist()]

    print("Datasets consolidated")
    return DataFrame(d)


def main(dir_: str = '') -> None:
    # Bike-Sharing-Dataset-hour is a detailed version of Bike-Sharing-Dataset-day, it will be used in process

    # Loads the BSS Configs
    dc_config = ml.Config(dir_, 'Bike-Sharing-Dataset-hour')
    london_config = ml.Config(dir_, 'london_merged-hour')

    # Loads the BSS datasets
    dc_dataset = ml.Dataset(dc_config.dataset)
    london_dataset = ml.Dataset(london_config.dataset)
    if not dc_dataset.load() or not london_dataset.load():
        return

    # loads the datasets
    dc_dataset.load()
    london_dataset.load()

    # handle missing data
    dc_dataset.apply(ml.handleMissingData)
    london_dataset.apply(ml.handleMissingData)

    # updates and saves the datasets with accompanying names
    # dc_dataset.update(name='Bike-Sharing-Dataset-hour-processed')
    # dc_dataset.save()
    # london_dataset.update(name='london-merged-hour-processed')
    # london_dataset.save()

    # data consolidation
    consolidated_bike = dataConsolidation(dc_dataset.df, london_dataset.df)

    bike_dataset = ml.Dataset(dc_config.dataset, df=consolidated_bike, name='bike-consolidated')
    bike_dataset.save()

    # feature selection
    # the selection can only remove the attributes which are totally depend on other attributes.
    bike_dataset.df.drop(["season", "is_workingday"], axis=1, inplace=True)
    bike_dataset.update(name='bike-consolidated-selected')
    bike_dataset.save()


if __name__ == '__main__':
    main(local_dir)
