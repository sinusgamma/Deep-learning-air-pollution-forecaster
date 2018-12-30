# Deep learning air-pollution forecaster
## for Udacity: PyTorch Scholarship Challenge from Facebook

Air pollution forecaster based on weather data and earlier pollution measurements.

## Project Idea

Numerical models are good at forecasting common weather parameters like wind or temperature, but they aren't so good at all weather-related problems. Air pollution is difficult to forecast because apart from the weather parameters we have to know the sources and sinks of the pollution during the forecast period. Most of the time these sources and sinks aren't well determined, and here a deep learning model can be really helpful, especially in cities, areas where we don't have any other information about air pollution than some air quality measurement device. Here deep learning can help to recognize the pattern of air pollution during the period, while the weather forecast model can calculate the wind, temperature, precipitation or other parameters that can be useful for the forecast. It is possible that a mixed deep-learning / numerical forecast model can be better than any of these models alone.

In this project, we build the deep learning part of the model.

## Data Explanation

The *'data\daily_clean_full.csv'* file containes the daily data of the weather parameters and air pollution (pm10) parameters. This is the cleaned merge of the *'daily_pm10.xlsx'* and the *'hourly_lorinc_2011-2018.xls'*
datasourcees:
https://rp5.ru/Weather_archive_in_Budapest,_Lorinc_(airport)
http://www.levegominoseg.hu/automata-merohalozat?city=2

*'daily_pm10.xlsx'*: measured PM10 in Budapest, 12 stations

*'hourly_lorinc_2011-2018.xls'*: Budapest Pestszentlőrinc weather station history - measurement

*'data\daily_clean_full.csv'* columns:<br />
**temp_avr**: average temperature based on hourly averages - ºC<br />
**temp_max**: maximum hourly temperature during the day - ºC<br />
**temp_min**: minimum hourly temperature during the day - ºC<br />
**pres**: mean sea level pressure - Hpa<br />
**u, v**: meridional and zonal component of wind - m/s<br />
**prec**: precipitation - mm (In stationdata we have 6h, 12h and 24h totals, and not hourly data. Some of the periods is missing. For simplicity in the cleaned data we use the largest of the above numbers, this way the order of magnitute of the precipitation remains, but the 24h total isn't precise. Later with better data this can be corrected.)<br />
**datetime**: datetime - UTC<br />
**pm10 station name**: pm10 concentration daily average - μg/m3<br />

(I suggest to use 2011, 2012, 2013, 2015, 2016, 2017 as training, 2014 as validation and 2018 as test. Other solutions can work, but we should use the same data to compare models.)



## Model Description
![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/first_model_idea.JPG)


## Authors and Contributors

* **xy** - github and/or linkedin link


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
Udacity, Facebook

