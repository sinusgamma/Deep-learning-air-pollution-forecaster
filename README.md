# Deep learning air-pollution forecaster
## Project for Udacity PyTorch Scholarship Challenge from Facebook

Air pollution PM10 forecaster based on weather data and earlier pollution measurements.

Side Project for: PyTorch Scholarship Challenge from Facebook
https://www.udacity.com/facebook-pytorch-scholarship

## Project Idea

Numerical models are good at forecasting common weather parameters like wind or temperature, but they aren't so good at all weather-related problems. Air pollution is difficult to forecast because apart from the weather parameters we have to know the sources and sinks of the pollution during the forecast period. Most of the time these sources and sinks aren't well determined. We hoped that an LSTM model can be used to predict the PM10 concentration. 

Our model will calculate only the PM10 concentration, while a weather forecast model can calculate the wind, temperature, precipitation or other parameters that can be the input of our model. For training we use station data for the weather parameter input instead of numerical prediction data.

We expect that the deep learning part of the model can help to grab the yearly, weekly or other patterns in air pollution fluctuation because of changes in traffic, heating or other effects.

We use weather information only from one weather station, and we use only the main parameters, so we don't expect very good performance, but this model can be a good basis for later improvements.

In this project, we build the deep learning part of the model.

This is a beginner project, we used codesnippets from the challenge course, especially from the character rnn forecast: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/recurrent-neural-networks/char-rnn

## Main Files

'pollution_forecaster.ipynb' and 'pollution_forecaster.py': 
You can train and load the model from these files, play with the parameters and check the performance.

'create_data.py': Prepares the daily data from the row datafiles.

'pollution_model.py': Containes the model and important functions to train it. 

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
**u, v**: meridional and zonal component of wind (daily averages) - m/s<br />
**prec**: precipitation - mm (In station data we have 6h, 12h and 24h totals, and not hourly data. Some of the periods are missing. For simplicity in the cleaned data we use the largest of the above numbers, this way the order of magnitude of the precipitation remains, but the 24h total isn't precise. Later with better data this can be corrected.)<br />
**datetime**: datetime - UTC<br />
**station name**: daily average pm10 concentration of station - μg/m3<br />

During training we used 2011-2016 for training, 2017 for validation and 2018 for the test.

We presumed some weekly pattern in air pollution concentration because of the traffic during the week, and yearly pattern because of the heating period, so we tried to help the model to learn this patterns by converting the day of the week and the day of the year to periodic form. This is detailed in pollution_forecaster.ipynb. But after training the model with and without these parameters we got only a very small performance improvement.

'create_data.py' was used to prepare the dataset used in the model.

### Simplifications
* Our model will use weather data only from one point in space. Because we try to forecast the pollution only in city level, we consider that point representative for the area. This point is our weather station.
* Instead of numerical weather forecast data we use station data for training. We don't use numerical forecast data for the training, because the numerical forecast can be wrong, and for training, we need 'perfect' weather prediction, because our model is responsible only for the air pollution prediction part.
* With the above simplifications, we use the station data as our forecast for training and validation. Reanalysis data could be better, later the station data can be replaced by that.
* Our forecast is valid only for the location of Teleki square airquality station.
* The trained model can use real weather forecast data to predict air pollution.
* One step in our model is one day. The daily average obscure the weather patterns during a day. An other model with hourly inputs could perform better.

### Model description

In time t we know the weather forecast for that time t, and we know the pollution concentration from time t-1. Our goal is to calculate the pollution concentration in time t. If we substitute the forecasted concentration in time t we can calculat the concentration for time t+1, and so on.

We used daily data, in our model one step is one day.

![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/image/base_model.JPG)


### Model Performance

A really good performance would have been a surprise because we used only one station as input, daily averages, and not all available weather parameters. What every forecast model should outperform is the prediction of yesterday. This means that we say every day, that we expect the same as yesterday. And this very simple forecast isn't a bad forecast.

In our test period during 2018 the 'yesterday=today' forecast would yield the following errors.

![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/image/error_noforecast.JPG)

Not so bad on average.

Our model was better, but the difference isn't very large:

![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/image/error_forecast.JPG)

We have to compare here the day0 errors with the 'yesterday=today' forecast errors.

The day1 errors show what happens if we input our pollution concentration forecast as input for the next day. It is not a surprise that for the next day our forecast will be worse.

![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/image/forecast_vs_act.JPG)

## How to improve?

There are lots of ways to improve the model performance. Our input data could be better for training, we could use weather reanalysis and hourly data instead of daily. We didn't use hyperparameter optimization. It is possible to use input and forecast over an area, not only a station. We could connect our model with real forecast and check what the model predicts for the next day.

### Model ideas for hourly steps

In the image below we can see two possible ways to use hourly input.

Model A:<br />
This is almost the same as our original model, but instead of daily steps it uses hourly steps.
This model could give us hourly concentration forecast.

Model B:<br />
Here our input would be hourly data, but our label would be daily average air pollution concentration for the next day.

Model A could predict every hour, but because every prediction would be an input of the next step, the error could be very large at the 24th step.
Model B could predict only the daily average concentration (or the hourly 24 hours later), but it wouldn't need the earlier prediction of the model in every step, the new information would be only the hourly weather, after the first step we could even omit the pollution concentration input. This way the model's own errors in the subsequent steps wouldn't affect the prediction.

There are other possible architectures which could improve the prediction, and it doesn't seem trivial which can be better. Best to try all that seems reasonable.

![alt text](https://github.com/sinusgamma/Deep-learning-air-pollution-forecaster/blob/master/image/modelAB.JPG)


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE.md](LICENSE.md) file for details


## Acknowledgments
Thanks for Facebook and Udacity for the chance to learn DL.