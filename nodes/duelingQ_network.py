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
        self.conv1 = nn.Conv2d(in_channels=IMAGE_HIST, out_channels=32, kernel_size=(10, 14))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        
        self.fc1 = nn.Linear(state_size, 512)
        init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        self.activation_fc1 = nn.Mish()
    
        self.fc2 = nn.Linear(200, 150)
        init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        self.activation_fc2 = nn.Mish()
    
        self.fc3 = nn.Dropout(0.1)
    
        self.fc4 = nn.Linear(150, 100)
        init.xavier_uniform_(self.fc4.weight, gain=nn.init.calculate_gain('relu'))
        self.activation_fc4 = nn.Mish()
    
        # self.fc5 = nn.Linear(9, action_size)
        # init.xavier_uniform_(self.fc5.weight, gain=nn.init.calculate_gain('linear'))
        # Values stream definition
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
        
    def forward(self, features):
        features = self.conv1(features)
        features = self.conv2(features)
        features = self.conv3(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.activation_fc1(features)
        features = self.fc2(features)
        features = self.activation_fc2(features)
        features = self.fc3(features)
        features = self.fc4(features)
        features = self.activation_fc4(features)
        Values_function = self.Values_stream(features)
        Advantages_function = self.Advantage_stream(features)
        Q_star = Values_function + (Advantages_function - Advantages_function.mean(dim=1, keepdim=True))
        return Q_star