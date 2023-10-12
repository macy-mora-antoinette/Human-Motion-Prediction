
import torch
from torch import nn
from TCN.tcn import TemporalConvNet
#from TCN.tcn import TemporalConvNet
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        #self.fc = torch.nn.Linear(120,24)
        self.fc = torch.nn.Linear(120,24)
        #self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        
        x= self.fc(x.transpose(1,2))
        
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        #output = self.tcn(x.transpose(1, 2)).transpose(1, 2).double()
        output = self.linear(output).double()
        return output
        #return self.sig(output)
