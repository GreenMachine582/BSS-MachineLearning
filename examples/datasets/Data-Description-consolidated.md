| Attribute     | Meaning                                                                       |
|---------------|-------------------------------------------------------------------------------|
| season        | category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter. |
| month         | month (1 to 12)                                                               |
| day           | day (1 to 28 or 29 or 30 or 31)                                               |
| hour          | hour (0 to 23)                                                                |
| is_holiday    | boolean field - 1 holiday / 0 non holiday                                     |
| is_weekend    | boolean field - 1 if the day is weekend                                       |
| is_workingday | boolean field - 1 if the day is workingday                                    |
| t1            | real temperature in C                                                         |
| t2            | temperature in C "feels like                                                  |
| wind_speed    | wind speed in km/h                                                            |
| weather       | category of the weather                                                       |
| city          | City where the data from - DC is short of Washington DC                       |
| cnt           | count of total rental bikes                                                   |
| rate          | number of people renting bicycles as a rate of the city's population          |
_rate: The values are derived via cnt*10000/city_population_

| Weather number | Weather                                                                                 |
|----------------|-----------------------------------------------------------------------------------------|
| 1              | Clear, Few clouds, Partly cloudy, Partly cloudy                                         |
| 2              | Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist                            |
| 3              | Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds |
| 4              | Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog                              |

