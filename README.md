# Deep learning air-pollution forecaster
## for Udacity: PyTorch Scholarship Challenge from Facebook

Air pollution forecaster based on weather data and earlier pollution measurements.

## Project Idea

Numerical models are good at forecasting common weather parameters like wind or temperature, but they aren't so good at all weather-related problems. Air pollution is difficult to forecast because apart from the weather parameters we have to know the sources and sinks of the pollution during the forecast period. Most of the time these sources and sinks aren't well determined, and here a deep learning model can be really helpful, especially in places where we don't have any other information about air pollution than some air quality measurement device. Here deep learning can help to recognize the pattern of air pollution during the period, while the weather forecast model can calculate the wind, temperature, precipitation or other parameters that can be useful for the forecast. It is possible that a mixed deep-learning / numerical forecast model can be better than any of these models alone. We expect that the deep learning part of the model can help to grab the yearly, weekly or other pattern in air pollution because of changes in traffic, heating or other effects. 

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

### Simplification
* Our model will use weather data only from one point in space. Because we try to forecast the pollution only in city level, we consider that point representative for the area. This point is our weather station.
* Instead of numerical weather forecast data we use station data for training. We don't use numerical forecast data for the training, because the numerical forecast can be wrong, and for training we need 'perfect' weather prediction, because our model is responsible only the air pollution prediction part.
* With the above simplifications we use the station data as our forecast for training and validation. Reanalysis data could be better, later the station data can be replaced by that.
* Our forecast is valid only for the location of given station or stations. It can't forecast to any coordinate.
* After training for real forecast naturally we can use real weather forecast data to predict air pollution for the future.

### Model A description

My basic idea:

In time t we know the weather forecast for that time t, and we know the pollution from time t-1. Our goal is to calculate the the pollution in t. 
So our **input** will be weather in time t and pollution in t-1
and our **label** is pollution in t.

In model A our label can be only 1 air quality station. We don't care about other stations.

After training we can run our model multiple times and use the forecast as input. This way we can forecast air pollution for multiple days.

![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/first_model_idea.JPG)


### Model B description

In model B in our input and label can be all air quality station. Maybe this helps to improve the model.

### Model C, D . . . description

Any idea what the group accepts.

#You are welcome to build model A, B, or any new idea.

## Authors and Contributors

* **xy** - github and/or linkedin link


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
Udacity, Facebook

