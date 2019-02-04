import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


# build the model for pollution forecast
class LSTMForecaster(nn.Module):
    
    def __init__(self, input_size, output_size, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001, train_on_gpu=False):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.train_on_gpu = train_on_gpu
        
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

        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


# instantiate model
# in our examination we don't want to use parameter tunning, so
# for simplicity only these four parameters are used
def instantiate_model(n_in, n_out, n_hidden, n_layers, train_on_gpu):
    '''instantiate LSTMforecaster'''
    net = LSTMForecaster(n_in, n_out, n_hidden, n_layers, train_on_gpu=train_on_gpu)
    return net


# Save the checkpoint
def save_checkpoint(net, val_loss, checkpoint_name):  
    checkpoint = {  'input_size': net.input_size,
                    'output_size': net.output_size,
                    'n_hidden': net.n_hidden,
                    'n_layers': net.n_layers,
                    'val_loss': val_loss,
                    'state_dict': net.state_dict()
                    }
    torch.save(checkpoint, checkpoint_name)


# train the model
def train(net, data_train, data_validation, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, checkpoint_name='checkpoint.pth', train_on_gpu=False):
    ''' Training a network   
        Arguments
        --------- 
        net: network
        data: data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        print_every: Number of steps for printing training and validation loss
    
    '''
    net.train()

    if(train_on_gpu):
        net.cuda()
    
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
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

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

            net.eval() # model to evaluation for validation process

            for inputs, targets in get_batches(data_validation, batch_size, seq_length):
                if(train_on_gpu):
                    inputs, targets = inputs.cuda(), targets.cuda()                
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output, targets.view(batch_size*seq_length,-1))
                val_losses.append(val_loss.item())   
            
            net.train() # reset to train mode after iterationg through validation data
            
            # save validation losses for later plotting
            mean_val_loss = np.mean(val_losses)
            mean_val_losses.append(mean_val_loss)

            # the next step are done only every fourth counter
            # if we would check every counter our model could be a bit 
            # better, but training would need more time
            if counter % 4 == 0:
                # check if our validation loss is lower than the earlier
                # and save the model if lower
                if mean_val_loss < min_val_loss:
                    min_val_loss = mean_val_loss
                    save_checkpoint(net, mean_val_loss, checkpoint_name)
                    print(f"model saved with {mean_val_loss} mean_val_loss") 
                # print some info
                print(  "Epoch: {}/{}...".format(e+1, epochs),
                        "Step: {}".format(counter),
                        "train_loss: {:.4f}...".format(train_loss.item()),
                        "mean_val_loss: {:.4f}".format(mean_val_loss))

            # print plot of loss         
            if counter % 16 == 0:        
                plt.plot(train_losses, label='Training loss')
                plt.plot(mean_val_losses, label='Validation loss')
                plt.legend(frameon=False)
                plt.show()