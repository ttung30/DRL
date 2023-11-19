#!usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=8, padding=1)
        init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.activation_conv1 = nn.Mish()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
        self.activation_conv2 = nn.Mish()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu'))
        
        self.activation_conv3 = nn.Mish()
        
        self.fc1 = nn.Linear(6336, 1024)
        init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        
        self.activation_fc1 = nn.Mish()

        # Lớp fully connected thứ hai
        self.fc2 = nn.Linear(1024, 512)
        init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        
        self.activation_fc2 = nn.Mish()

        # Lớp fully connected thứ ba
        self.fc3 = nn.Linear(512, 100)
        init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))
        
        self.activation_fc3 = nn.Mish()
        
        self.Values_stream = nn.Sequential(
            nn.Linear(100, 50),
            nn.Mish(),
            nn.Linear(50, 1)
        )
        # Advantage stream definition
        self.Advantage_stream = nn.Sequential(
            nn.Linear(100, 50),
            nn.Mish(),
            nn.Linear(50, action_size)
        )
        
    def forward(self, feature):
        feature = self.conv1(feature)
        feature = self.activation_conv1(feature)
        
        
        # Kích thước sau lớp tích chập thứ nhất: (batch_size, 32, 16, 16)
        feature = self.conv2(feature)
        feature = self.activation_conv2(feature)
        
        
        feature = self.conv3(feature)
        feature = self.activation_conv3(feature)
        
        
        feature = feature.view(feature.size(0), -1) 
        feature = self.fc1(feature)
        feature = self.activation_fc1(feature)

        feature = self.fc2(feature)
        feature = self.activation_fc2(feature)
        feature = self.fc3(feature)
        feature = self.activation_fc3(feature)
        Values_function = self.Values_stream(feature)
        Advantages_function = self.Advantage_stream(feature)
      
        Q_star = Values_function + (Advantages_function - Advantages_function.mean(dim=1, keepdim=True))
       
        return Q_star