#%% [markdown]
# # Deep Learning LSTM PM10 air pollution forecaster
# This notebook loads the data, instantiates the LSTM model, and trains the model, then calculates the predictions.
# 
# The 'pollution_model.py' file containes the model and the functions to instantiate, train, save and run the model.

#%%
# ### Import resources and create data
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import pollution_model as pmod
import warnings
warnings.filterwarnings('ignore')

#%% [markdown]
# The data containes the daily average temperature, maximum, minimum temperature, sea level preassure, wind converted to u and v parameters, precipitation and PM10 concentration for the day. This is calculated from hourly parameters with the 'create_data.py' file.

#%%
# load data
df_full = pd.read_csv('daily_clean_full.csv')
df_full['datetime'] = pd.to_datetime(df_full['datetime'], format='%Y-%m-%d')
df_full.head()

#%% [markdown]
# We use only one station for our forecast: Budapest, **Teleki** square

#%%
# get data for single station forecast
df = df_full[['datetime', 'temp_avr', 'temp_max', 'temp_min', 'pres', 
              'u', 'v', 'prec', 'Teleki']]
df.head()

#%% [markdown]
# We need to normalize the data before training. For the air pollution columns we save the normalization parameters. Later with them we can convert the forecasted outputs to pollution concentrations.

#%%
# variables for later renormalization
label_mean = df['Teleki'].mean()
label_std = df['Teleki'].std()
print(f"pollution columns mean: {label_mean}")
print(f"pollution columns std: {label_std}")
#%%
# normalize columns with (c-mean)/std
df[['temp_avr','temp_max','temp_min','pres','u','v','prec','Teleki']] = df[['temp_avr','temp_max','temp_min','pres','u','v','prec','Teleki']] .apply(lambda x: (x - x.mean()) / x.std())

df.head()

#%% [markdown]
# We convert the day of week and year of week to coordinates on a one radius circel. This way the days closer to each other will have similar coordinates. We hope that this helps the model to understand some periodic behavior of the dataset.

#%%
# Convert day of year and day of week numbers to coordinates of a point
# on a circle. This way day of year and day of week will become periodic 
# data
# first convert days to scale
df['day_week'] = df['datetime'].dt.dayofweek*(360/7)
df['day_year'] = df['datetime'].dt.dayofyear*(360/365)
# then convert scales to coordinates on 1 radius circle
# rad = 4.0*atan(1.0)/180
df['dw_x'] = -np.sin((4.0*np.arctan(1.0)/180)*df['day_week'])
df['dw_y'] = -np.cos((4.0*np.arctan(1.0)/180)*df['day_week'])
df['dy_x'] = -np.sin((4.0*np.arctan(1.0)/180)*df['day_year'])
df['dy_y'] = -np.cos((4.0*np.arctan(1.0)/180)*df['day_year'])

df.plot.scatter(x='dy_x', y='dy_y')
df.plot.scatter(x='dw_x', y='dw_y')

#%% [markdown]
# In real situation for a given day we know the weather (from a weather forecast, which here is represented by the weather station data) and the airpollution a day before (Teleki_ystd). So we add to a row of parameters the airpollution of the prior day. The air pollution of the given day (Teleki) will be our label. We want our forecast as close to this label as possible.

#%%
# make the yesterday day column
df['Teleki_ystd'] = df['Teleki'].shift(+1)
# fill the missing day value with next day
df['Teleki_ystd'] = df['Teleki_ystd'].fillna(method='bfill')
# reorder columns, last column will be the label data
df = df[['datetime','dy_x','dy_y','dw_x','dw_y',
         'temp_avr','temp_max','temp_min','pres', 
         'u','v','prec','Teleki_ystd','Teleki']]
df.head()

#%% [markdown]
# Split the database to train, validation and test data.

#%%
# build the pytorch train, validation and test sets
# train data
mask = (df['datetime'] < '2017-01-01')
df_train = df.loc[mask].drop(columns=['datetime'])
# train tensors
train_data = torch.tensor(df_train.values).float()
print(train_data.shape)

# validation data
mask = (df['datetime'] < '2018-01-01') & (df['datetime'] >= '2017-01-01')
df_valid = df.loc[mask].drop(columns=['datetime'])
# validation tensors
valid_data = torch.tensor(df_valid.values).float()
print(valid_data.shape)

# test data
mask = (df['datetime'] >= '2018-01-01')
df_test = df.loc[mask].drop(columns=['datetime'])
# validation tensors
test_data = torch.tensor(df_test.values).float()
print(test_data.shape)


#%%
# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')

#%% [markdown]
# Here we instantiate our model.

#%%
# define and print the net
number_of_features = train_data.shape[1] - 1 # -1 because last column is label
n_hidden=256
n_layers=3
net = pmod.instantiate_model(number_of_features, 1, n_hidden, n_layers, train_on_gpu)
print(net)

#%% [markdown]
# We train our model. During training the step with the smallest validation loss will be saved. Later we will load back the modell with the smallest loss.

#%%
# train the model
cp_name='cp_withtimeperiod.pth'
batch_size = 5
seq_length = 30
n_epochs = 100 # start smaller if you are just testing initial behavior
pmod.train( net, train_data, valid_data, epochs=n_epochs, 
            batch_size=batch_size, seq_length=seq_length, lr=0.001, 
            checkpoint_name=cp_name, train_on_gpu=train_on_gpu)


#%%
# load back best model
with open(cp_name, 'rb') as f:
    checkpoint = torch.load(f)
  
net_best = pmod.instantiate_model(checkpoint['input_size'], checkpoint['output_size'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'], train_on_gpu=train_on_gpu)
net_best.load_state_dict(checkpoint['state_dict'])
if(train_on_gpu):
  net_best.cuda()

#%% [markdown]
# Use the model with test_data to forecast the first day pollution (day0) and the second day pollution (day1). For the forecast of the second day pollution we use our own forecast of first day pollution.

#%%
# check test data
# model to evaluation mode
net_best.eval()
test_losses_MSE = [[], []]
target_list = [[], []]
forecast_list = [[], []]
# during test we simulate the implementation of the deployed usage:
batch_size = 1
seq_length = 1

criterionMSE = nn.MSELoss()

counter = 0
# we will examine the model forecast for the next day where the
# pollution is unknown and for the they after that
# we implement two data-getter for that, one for the first forecast day
# and an other for the next forecast day
batches_day0 = pmod.get_batches(test_data, batch_size, seq_length)
batches_day1 = pmod.get_batches(test_data, batch_size, seq_length)
next(batches_day1)

# data for first forecast day
for inputs_day0, targets_day0 in batches_day0:
    counter+=1
    # break before batches_day1 runs out
    if counter >= len(test_data):
        break
    # data for second forecast day    
    inputs_day1, targets_day1 = next(batches_day1)
    if(train_on_gpu):
      inputs_day0, targets_day0 = inputs_day0.cuda(), targets_day0.cuda()
      inputs_day1, targets_day1 = inputs_day1.cuda(), targets_day1.cuda()

    # calculate forecast for day0
    test_h = net.init_hidden(batch_size)
    output_day0, test_h = net_best(inputs_day0, test_h)
    test_loss_MSE_day0 = criterionMSE(output_day0, targets_day0.view(batch_size*seq_length,-1))
    test_losses_MSE[0].append(test_loss_MSE_day0.item())
    target_list[0].append(targets_day0.item())
    forecast_list[0].append(output_day0.item())

    # need to clone tensor, without this the modification of tensor would affect other data as well
    inputs_day1.data = inputs_day1.clone()
    # we don't change the weather parameters for inputs_day1, because that is our weather forecast,
    # but we change the last number, because that is our pollution forecast from day0
    # our beforday pollution in day1 is the pollution forecast of day0
    inputs_day1[0][0][-1] = output_day0.item()
    # calculate forecast for the day1
    test_h = net.init_hidden(batch_size)
    output_day1, test_h = net_best(inputs_day1, test_h)
    test_loss_MSE_day1 = criterionMSE(output_day1, targets_day1.view(batch_size*seq_length,-1))
    test_losses_MSE[1].append(test_loss_MSE_day1.item())
    target_list[1].append(targets_day1.item())
    forecast_list[1].append(output_day1.item())

print( "test losses MSE for day0: {:.4f}".format(np.mean(test_losses_MSE[0])))
print( "test losses MSE for day1: {:.4f}".format(np.mean(test_losses_MSE[1])))

#%% [markdown]
# To examine the forecast performance we convert back our data to the original form. This way we get back the μg/m3 concentration numbers. 

#%%
# renormalize data to get back the pm10 concentrations
forecast_concentration_day0 = [i * label_std + label_mean for i in forecast_list[0]]
target_concentration_day0 = [i * label_std + label_mean for i in target_list[0]]
abs_error_day0 = [np.abs(a - b) for a, b in zip(forecast_concentration_day0, target_concentration_day0)]

forecast_concentration_day1 = [i * label_std + label_mean for i in forecast_list[1]]
target_concentration_day1 = [i * label_std + label_mean for i in target_list[1]]
abs_error_day1 = [np.abs(a - b) for a, b in zip(forecast_concentration_day1, target_concentration_day1)]

#%% [markdown]
# The most simple forecast is if we say that something tomorrow will be the same as something today. We hope that our model is better than that. First calculat the errors for this 'noforecast' forecast, only for day0.

#%%
abs_error_day0_noforecast = [np.abs(a - b) for a, b in zip(target_concentration_day1, target_concentration_day0)]
print("ERRORS OF NO FORECAST - we except the same as yesterday")
print(f"Mean Abs Error no forecast: {np.mean(abs_error_day0_noforecast)} μg/m3")
print(f"Median Abs Error no forecast: {np.median(abs_error_day0_noforecast)} μg/m3")
print(f"Max Abs Error no forecast: {np.max(abs_error_day0_noforecast)} μg/m3")

#%% [markdown]
# Now we calculate our forecasts by the model for day0 and the next day1.

#%%
# error stat day0s:
print("ERRORS OF DAY0 FORECAST")
print(f"Mean Abs Error day0: {np.mean(abs_error_day0)} μg/m3")
print(f"Median Abs Error day0: {np.median(abs_error_day0)} μg/m3")
print(f"Max Abs Error day0: {np.max(abs_error_day0)} μg/m3")
print("MSE for day0: {:.4f}".format(np.mean(test_losses_MSE[0])))

# error stats day1:
print("ERRORS OF DAY1 FORECAST")
print(f"Mean Abs Error day1: {np.mean(abs_error_day1)} μg/m3")
print(f"Median Abs Error day1: {np.median(abs_error_day1)} μg/m3")
print(f"Max Abs Error day1: {np.max(abs_error_day1)} μg/m3")
print("MSE for day1: {:.4f}".format(np.mean(test_losses_MSE[1])))


#%%
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].plot(forecast_concentration_day0, label='forecast')
axs[0].plot(target_concentration_day0, label='actual')
axs[0].plot(abs_error_day0, label='abs error')
axs[0].set_title('day0 forecast')
axs[0].set_xlabel('day of year')
axs[0].set_ylabel('μg/m3')
axs[0].legend()
fig.suptitle('Forecast and measurement during the test period', fontsize=16)

axs[1].plot(forecast_concentration_day1, label='forecast')
axs[1].plot(target_concentration_day1, label='actual')
axs[1].plot(abs_error_day1, label='abs error')
axs[1].set_title('day1 forecast')
axs[1].set_xlabel('day of year')
axs[1].set_ylabel('μg/m3')
axs[1].legend()

fig.subplots_adjust(hspace=0.3)

plt.show()


