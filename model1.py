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
# normalize columns with (c-mean)/std
df[['temp_avr',
    'temp_max', 
    'temp_min', 
    'pres', 
    'u', 
    'v',
    'prec',
    'Teleki']] = StandardScaler().fit_transform(df[[  'temp_avr',
                                                        'temp_max', 
                                                        'temp_min', 
                                                        'pres', 
                                                        'u', 
                                                        'v',
                                                        'prec',
                                                        'Teleki']])
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
df_train_input = df_train.drop(columns=['Teleki'])
df_train_label = df_train['Teleki']
df_train_label
# train tensors
train_data = torch.tensor(df_train.values).float()
train_input = torch.tensor(df_train_input.values).float()
train_label = torch.tensor(df_train_label.values).float()
train_data

#%%
# validation data
mask = (df['datetime'] < '2018-01-01') & (df['datetime'] >= '2017-01-01')
df_valid = df.loc[mask].drop(columns=['datetime'])
df_valid_input = df_valid.drop(columns=['Teleki'])
df_valid_label = df_valid['Teleki']
df_valid_label
# validation tensors
valid_data = torch.tensor(df_valid.values).float()
valid_input = torch.tensor(df_valid_input.values).float()
valid_label = torch.tensor(df_valid_label.values).float()
valid_data

#%%
# test data
mask = (df['datetime'] >= '2018-01-01')
df_test = df.loc[mask].drop(columns=['datetime'])
df_test_input = df_test.drop(columns=['Teleki'])
df_test_label = df_test['Teleki']
df_test_label
# validation tensors
test_data = torch.tensor(df_test.values).float()
test_input = torch.tensor(df_test_input.values).float()
test_label = torch.tensor(df_test_label.values).float()

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
    for n in range(0, batch_size, arr.shape[0]):
        # The features
        x= arr[n:n+batch_size,:,:-1]
        # The targets
        y = y= arr[n:n+batch_size,:,-1]
        yield x, y

#%%
# test get_batches function
batches = get_batches(train_data, batch_size=2, seq_length=3)
x, y = next(batches)
print('x\n', x)
print(x.shape)
print('\ny\n', y)
print(y.shape)

#%%
# build the model
class LSTMForecaster(nn.Module):
    
    def __init__(self, input_size, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.input_size
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        ## TODO: define the LSTM
        self.lstm = nn.LSTM(input_size, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, 1)
      
    
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
# not tested yet
# Save the checkpoint
def save_checkpoint(net, loss):
  checkpoint = {    'n_inputs': net.input_size,
                    'n_hidden': net.n_hidden,
                    'n_layers': net.n_layers,
                    'state_dict': net.state_dict(),
                    'loss' : loss} 
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
            loss = criterion(output, targets.view(batch_size*seq_length,-1))
            train_losses.append(loss.item())
            loss.backward()
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
            
            print("Epoch: {}/{}...".format(e+1, epochs),
                    "Loss: {:.4f}...".format(loss.item()),
                    "Val Loss: {:.4f}".format(np.mean(val_losses)))
                    no_loss_decrease_steps += 1
            # not tested yet        
            '''if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_checkpoint(net, val_loss)
                print(f"model saved with {val_loss)} validation loss")'''        
            # print plot of loss         
            if counter % 5 == 0:        
                plt.plot(train_losses, label='Training loss')
                plt.plot(val_losses, label='Validation loss')
                plt.legend(frameon=False)
                plt.show()      


#%%
# define and print the net
n_hidden=512
n_layers=2
net = LSTMForecaster(number_of_features, n_hidden, n_layers)
print(net)

#%%
batch_size = 2
seq_length = 30
n_epochs = 100 # start smaller if you are just testing initial behavior

# train the model
train(net, train_data, valid_data, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001)               
