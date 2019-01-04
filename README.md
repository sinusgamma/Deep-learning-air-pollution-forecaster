# Deep learning air-pollution forecaster
## project for Udacity PyTorch Scholarship Challenge from Facebook

Air pollution forecaster based on weather data and earlier pollution measurements.

## Project Idea

Numerical models are good at forecasting common weather parameters like wind or temperature, but they aren't so good at all weather-related problems. Air pollution is difficult to forecast because apart from the weather parameters we have to know the sources and sinks of the pollution during the forecast period. Most of the time these sources and sinks aren't well determined. We hoped that an LSTM model can be used to predict the PM10 concentration. 

Our model will calculate only the PM10 concentration, while a weather forecast model can calculate the wind, temperature, precipitation or other parameters that can be the input of our model. For training instead of numerical model data we use station data for the weather parameter input.

We expect that the deep learning part of the model can help to grab the yearly, weekly or other patterns in air pollution fluctuation because of changes in traffic, heating or other effects.

We use weather information only from one weather station, and we use only the main parameters, so we don't expect very good performance, but this model can be a good basis for later improvements.

In this project, we build the deep learning part of the model.

## Data Explanation

The *'data\daily_clean_full.csv'* file contains the daily data of the weather parameters and air pollution (pm10) parameters. This is the cleaned merge of the *'daily_pm10.xlsx'* and the *'hourly_lorinc_2011-2018.xls'*
datasources:
https://rp5.ru/Weather_archive_in_Budapest,_Lorinc_(airport)
http://www.levegominoseg.hu/automata-merohalozat?city=2

*'daily_pm10.xlsx'*: measured PM10 in Budapest, 12 stations
We made forecast only for one point: Budapest, Teleki square airquality station.

*'hourly_lorinc_2011-2018.xls'*: Budapest Pestszentlőrinc weather station history - measurement

*'data\daily_clean_full.csv'* columns:<br />
**temp_avr**: average temperature based on hourly averages - ºC<br />
**temp_max**: maximum hourly temperature during the day - ºC<br />
**temp_min**: minimum hourly temperature during the day - ºC<br />
**pres**: mean sea level pressure - Hpa<br />
**u, v**: meridional and zonal component of wind - m/s<br />
**prec**: precipitation - mm (In station data we have 6h, 12h and 24h totals, and not hourly data. Some of the periods are missing. For simplicity in the cleaned data we use the largest of the above numbers, this way the order of magnitude of the precipitation remains, but the 24h total isn't precise. Later with better data this can be corrected.)<br />
**datetime**: datetime - UTC<br />
**pm10 station name**: pm10 concentration daily average - μg/m3<br />

During training we used 2011-2016 for training, 2017 for validation and 2018 for the test.

We presumed some weekly pattern in air pollution concentration because of the traffic during the week, and yearly pattern because of the heating period, so we tried to help the model to learn this patterns by converting the day of the week and the day of the year to periodic form. This is detailed in pollution_forecaster.ipynb. But after training the model with and without these parameters we got only a very small performance improvement.

'create_data.py' was used to prepare the dataset used in the model.

### Simplifications
* Our model will use weather data only from one point in space. Because we try to forecast the pollution only in city level, we consider that point representative for the area. This point is our weather station.
* Instead of numerical weather forecast data we use station data for training. We don't use numerical forecast data for the training, because the numerical forecast can be wrong, and for training, we need 'perfect' weather prediction, because our model is responsible only for the air pollution prediction part.
* With the above simplifications, we use the station data as our forecast for training and validation. Reanalysis data could be better, later the station data can be replaced by that.
* Our forecast is valid only for the location of Teleki square airquality station.
* The trained model can be used real weather forecast data to predict air pollution.

### Model description

In time t we know the weather forecast for that time t, and we know the pollution concentration from time t-1. Our goal is to calculate the pollution concentration in time t. If we substitute the forecasted concentration in time t we can calculat the concentration in time t+1 . . . 

We used daily data, in our model one step is one day.

![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/image/first_model_idea.JPG)


### Model Performance

A really good performance would have been a surprise because we used only one station as input, and not all available weather parameters. What every forecast model must outperform is the prediction of yesterday. This means that we say every day, that we expect the same as yesterday. And this isn't a really bad model.

In our test period during 2018 the 'yesterday=today' forecast would yield the following errors.

![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/image/error_noforecast.JPG)

Not so bad on average.

Our model was better, but the difference isn't very large:

![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/image/error_forecast.JPG)

We have to compare here the day0 errors with the 'yesterday=today' forecast errors.

The day1 errors show what happens if we input our pollution concentration forecast as input for the next day. It is not a surprise that for the next day our forecast will be worse.

![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/image/forecast_vs_act.JPG)

## How to improve?

There are lots of ways to improve the model. Our input data could be better, we didn't use hyperparameter optimization, it is possible to use input and forecast over an area, not only a station, or inputting data from real forecast and check what the model predicts for the next day.


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE.md](LICENSE.md) file for details


## Acknowledgments
Thanks for Facebook and Udacity for the chance to learn DL.

