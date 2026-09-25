import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from PE_Data import get_dat
from PE_Data import prep_dat

file = "Buyout_Funds_Stats_2014.xlsx"

dat = get_dat(file, "Distributed", in_between=90)

# Hyperparameters
num_epochs = 100
learning_rate = 0.005
input_size = 1
batch_size = 100 
hidden_size = 270 #historical sequence length, in days
num_train = 100

# 0 index corresponds to benchmark of all sizes
sup_dat = prep_dat(dat[0], n_step=hidden_size, n_ahead=360)

# Split up data, 70-30 for training and test set
X_train = sup_dat.iloc[0:math.floor(0.7 * len(sup_dat)), 0:hidden_size]
y_train = sup_dat.iloc[0:math.floor(0.7 * len(sup_dat)), hidden_size]
X_test = sup_dat.iloc[math.floor(0.7 * len(sup_dat)):len(sup_dat), 0:hidden_size]
y_test = sup_dat.iloc[math.floor(0.7 * len(sup_dat)):len(sup_dat), hidden_size]


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=1):
        
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers) 
        
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, data_in, batch):
        
        # Forward pass through LSTM layer
        row = data_in.shape[0]
        col = data_in.shape[1]
        data_in = data_in.view(col, row, 1)
        lstm_out, self.hidden = self.lstm(data_in)
        y_pred = self.linear(lstm_out[-1].view(row, -1))
        return y_pred.view(-1)

def trainModel(model, x_train, y_train, c):
    
    total_loss = 0
    y = np.array([])
    
    for i in range(math.floor(len(x_train)/batch_size)+1): #### +1
        
        try:
            x_data = torch.tensor(x_train.iloc[i * batch_size:(i + 1) * batch_size].values).float()
            y_data = torch.tensor(y_train.iloc[i * batch_size:(i + 1) * batch_size].values).float()
        
        except:
            # last batch
            x_data = torch.tensor(x_train.iloc[i*batch_size:].values).float()
            y_data = torch.tensor(y_train.iloc[i*batch_size:].values).float()

        x_data = norm_batch(x_data)
        model.zero_grad()
        
        # Forward pass
        y_pred = model.forward(x_data, batch_size)
        loss = loss_fn(y_pred, y_data)
        
        # Backward and optimize
        optimiser.zero_grad()
        loss.backward()  # Backward pass
        optimiser.step()  # Update parameters
        y = np.append(y, y_pred.detach().numpy())
        total_loss += loss.item()

    print('train set length:', len(x_train))
    avg_loss = total_loss / math.floor(len(x_train) / batch_size)
    print("Epoch ", c, "MSE: ", avg_loss)
    
    return model, total_loss / len(x_train)


def testModel(model, x_test, y_test):
    
    total_loss = 0
    model.eval()
    y = np.array([])
    
    for i in range(math.floor(len(x_test) / batch_size)+1):
        
        #feed in test data by batches
        try:
            x_data = torch.tensor(x_test.iloc[i * batch_size:(i + 1) * batch_size].values).float()
            y_data = torch.tensor(y_test.iloc[i * batch_size:(i + 1) * batch_size].values).float()
        
        except:
            x_data = torch.tensor(x_test.iloc[i * batch_size:].values).float()
            y_data = torch.tensor(y_test.iloc[i * batch_size:].values).float()
        
        x_data = norm_batch(x_data)
        
        # Forward pass
        y_pred = model.forward(x_data, batch_size)
        loss = loss_fn(y_pred, y_data)
        y = np.append(y, y_pred.detach().numpy())
        total_loss += loss.item()
        
    #plot predicted vs actual test values
    plt.plot(y_test.values, label='training')
    plt.plot(y, label='pred')
    plt.title('Testing Set')
    plt.legend()
    plt.show()
    
    avg_loss = total_loss / math.floor(len(x_test) / batch_size)
    print("MSE test: ", avg_loss)

    return model, total_loss / len(x_test)

# Declare architecture
model = LSTM(input_size, hidden_size, batch_size=num_train, num_layers=1)
norm_batch = nn.BatchNorm1d(hidden_size)
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
hist = np.zeros(num_epochs)

# Train
for j in range(num_epochs):
    model, total_loss = trainModel(model, X_train, y_train, j + 1)
    
print("Optimization finished.")

# Test
model, loss = testModel(model, X_test, y_test)
