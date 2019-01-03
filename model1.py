#%% [markdown]
# # Model 1 

#%%
# ### Import resources and create data
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from sklearn.preprocessing import StandardScaler

#%%
# load data
df_full = pd.read_csv('data\daily_clean_full.csv')
df_full['datetime'] = pd.to_datetime(df_full['datetime'], format='%Y-%m-%d')
df_full.head()

#%%
# get data for single station forecast
df = df_full[[  'datetime',
                'temp_avr', 
                'temp_max', 
                'temp_min', 
                'pres', 
                'u', 
                'v',
                'prec',
                'Teleki'
                 ]]
df.head()

#%%
# variables for later renormalization
label_mean = df['Teleki'].mean()
label_std = df['Teleki'].std()
print(label_mean)
print(label_std)
# normalize columns with (c-mean)/std
df[['temp_avr', 
    'temp_max', 
    'temp_min', 
    'pres', 
    'u', 
    'v',
    'prec',
    'Teleki']]=df[[ 'temp_avr', 
                    'temp_max', 
                    'temp_min', 
                    'pres', 
                    'u', 
                    'v',
                    'prec',
                    'Teleki']].apply(lambda x: (x - x.mean()) / x.std())
df.head()

#%%
# make the yesterday day column
df['Teleki_ystd'] = df['Teleki'].shift(+1)
# fill the missing day value with next day
df['Teleki_ystd'] = df['Teleki_ystd'].fillna(method='bfill')
# reorder columns, last column will be the label data
df = df[[   'datetime',
            'temp_avr', 
            'temp_max', 
            'temp_min', 
            'pres', 
            'u', 
            'v',
            'prec',
            'Teleki_ystd',
            'Teleki']]
df.head()

#%%
# build the pytorch train, validation and test sets
# train data
mask = (df['datetime'] < '2017-01-01')
df_train = df.loc[mask].drop(columns=['datetime'])
# train tensors
train_data = torch.tensor(df_train.values).float()
print(train_data.shape)
train_data

#%%
# validation data
mask = (df['datetime'] < '2018-01-01') & (df['datetime'] >= '2017-01-01')
df_valid = df.loc[mask].drop(columns=['datetime'])
# validation tensors
valid_data = torch.tensor(df_valid.values).float()
print(valid_data.shape)
valid_data

#%%
# test data
mask = (df['datetime'] >= '2018-01-01')
df_test = df.loc[mask].drop(columns=['datetime'])
# validation tensors
test_data = torch.tensor(df_test.values).float()
print(test_data.shape)
test_data

#%%
# the function to yield the batches
def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''
    feature_number = arr.shape[1]
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total 
    # Keep only enough days to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows and feature_number vectors
    arr = arr.reshape((-1, seq_length, feature_number))
      
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[0], batch_size):
        # The features
        x= arr[n:n+batch_size,:,:-1]
        # The targets
        y = y= arr[n:n+batch_size,:,-1]
        yield x, y

#%%
# test get_batches function
batches = get_batches(test_data, batch_size=2, seq_length=30)
#%%
x, y = next(batches)
print('x\n', x)
print(x.shape)
print('\ny\n', y)
print(y.shape)

#%%
# build the model
class LSTMForecaster(nn.Module):
    
    def __init__(self, input_size, output_size, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        
        ## TODO: define the LSTM
        self.lstm = nn.LSTM(input_size, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, output_size)
      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)     
        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden) 
        # put x through the fully-connected layer
        out = self.fc(out)
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_()) 
        return hidden

#%%
# Save the checkpoint
def save_checkpoint(net, val_loss):  
    checkpoint = {  'input_size': net.input_size,
                    'output_size': net.output_size,
                    'n_hidden': net.n_hidden,
                    'n_layers': net.n_layers,
                    'val_loss': val_loss,
                    'state_dict': net.state_dict()
                    }
    torch.save(checkpoint, 'checkpoint.pth')

#%%
def train(net, data_train, data_validation, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5):
    ''' Training a network   
        Arguments
        --------- 
        net: CharRNN network
        data: data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        print_every: Number of steps for printing training and validation loss
    
    '''
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
  
    counter = 0
    train_losses = []
    val_losses = []
    mean_val_losses = []
    min_val_loss = np.Inf
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        for inputs, targets in get_batches(data_train, batch_size, seq_length):
            counter += 1
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs, h)
            
            # calculate the loss and perform backprop
            train_loss = criterion(output, targets.view(batch_size*seq_length,-1))
            train_losses.append(train_loss.item())
            train_loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            # Get validation loss
            val_h = net.init_hidden(batch_size)

            net.eval()

            for inputs, targets in get_batches(valid_data, batch_size, seq_length):
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output, targets.view(batch_size*seq_length,-1))
                val_losses.append(val_loss.item())   
            
            net.train() # reset to train mode after iterationg through validation data
            
            mean_val_loss = np.mean(val_losses)
            mean_val_losses.append(mean_val_loss)

            if mean_val_loss < min_val_loss:
                min_val_loss = mean_val_loss
                save_checkpoint(net, mean_val_loss)
                print(f"model saved with {mean_val_loss} mean_val_loss") 

            print(  "Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}".format(counter),
                    "train_loss: {:.4f}...".format(train_loss.item()),
                    "mean_val_loss: {:.4f}".format(mean_val_loss))

            # print plot of loss         
            if counter % 10 == 0:        
                plt.plot(train_losses, label='Training loss')
                plt.plot(mean_val_losses, label='Validation loss')
                plt.legend(frameon=False)
                plt.show()

#%%
# define and print the net
number_of_features = train_data.shape[1] - 1 # -1 because last column is label
n_hidden=512
n_layers=2
net = LSTMForecaster(number_of_features, 1, n_hidden, n_layers)
print(net)

#%%
# train the model
#############################
batch_size = 10
seq_length = 7
#############################
n_epochs = 10 # start smaller if you are just testing initial behavior
train(net, train_data, valid_data, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001)               


#%%
# load back best model
with open('checkpoint.pth', 'rb') as f:
    checkpoint = torch.load(f)
    
net_best = LSTMForecaster(checkpoint['input_size'], checkpoint['output_size'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
net_best.load_state_dict(checkpoint['state_dict'])

#%%
# check test data
net_best.eval()
test_losses_MSE = [[], []]
target_list = [[], []]
forecast_list = [[], []]
batch_size = 1
seq_length = 1

criterionMSE = nn.MSELoss()

counter = 0
batches_day0 = get_batches(test_data, batch_size, seq_length)
batches_day1 = get_batches(test_data, batch_size, seq_length)
next(batches_day1)

for inputs_day0, targets_day0 in batches_day0:
    counter+=1
    # break before batches_day1 runs out
    if counter >= len(test_data):
        break
    inputs_day1, targets_day1 = next(batches_day1)

    # calculate forecast for day0
    test_h = net.init_hidden(batch_size)
    output_day0, test_h = net_best(inputs_day0, test_h)
    test_loss_MSE_day0 = criterionMSE(output_day0, targets_day0.view(batch_size*seq_length,-1))
    test_losses_MSE[0].append(test_loss_MSE_day0.item())
    target_list[0].append(targets_day0.item())
    forecast_list[0].append(output_day0.item())

    # need to clone tensor, without this the modification would affect other part as well
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

#%%
# renormalize data to get back the pm10 concentrations
forecast_concentration_day0 = [i * label_std + label_mean for i in forecast_list[0]]
target_concentration_day0 = [i * label_std + label_mean for i in target_list[0]]
abs_error_day0 = [np.abs(a - b) for a, b in zip(forecast_concentration_day0, target_concentration_day0)]

#%%
# renormalize data to get back the pm10 concentrations
forecast_concentration_day1 = [i * label_std + label_mean for i in forecast_list[1]]
target_concentration_day1 = [i * label_std + label_mean for i in target_list[1]]
abs_error_day1 = [np.abs(a - b) for a, b in zip(forecast_concentration_day1, target_concentration_day1)]

#%%
# plot forecast for day0
plt.plot(forecast_concentration_day0, label='forecast')
plt.plot(target_concentration_day0, label='actual')
plt.plot(abs_error_day0, label='abs error')
plt.legend(frameon=False)
plt.show()
# error stats:
print(f"Mean Abs Error day0: {np.mean(abs_error_day0)}")
print(f"Median Abs Error day0: {np.median(abs_error_day0)}")
print(f"Max Abs Error day0: {np.max(abs_error_day0)}")

#%%
# plot forecast for day0
plt.plot(forecast_concentration_day1, label='forecast')
plt.plot(target_concentration_day1, label='actual')
plt.plot(abs_error_day1, label='abs error')
plt.legend(frameon=False)
plt.show()
# error stats:
print(f"Mean Abs Error day1: {np.mean(abs_error_day1)}")
print(f"Median Abs Error day1: {np.median(abs_error_day1)}")
print(f"Max Abs Error day1: {np.max(abs_error_day1)}")

#%%
batches_day0 = get_batches(test_data, batch_size, seq_length)
batches_tomorrow = get_batches(test_data, batch_size, seq_length)
next(batches_tomorrow)
#%%
inputs_today, targets_today = next(batches_day0)
print(inputs_today)
inputs_tomorrow, targets_tomorrow = next(batches_tomorrow)
print(inputs_tomorrow)
inputs_tomorrow[0][0][-1] = 6
print(inputs_tomorrow)

#%%



