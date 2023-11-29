#!usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
import rospy
import random
import numpy as np
import os
#from torchvision.transforms import functional as Fa
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from duelingQ_network import DuelingQNetwork

LOSS_DATA_DIR = os.path.dirname(os.path.realpath(__file__))
LOSS_DATA_DIR = LOSS_DATA_DIR.replace('dueling_dqn_gazebo/nodes', 'dueling_dqn_gazebo/save_model/Dueling_DQN_trial_1/log_data')

class MemoryBuffer():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        
    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        states_array = np.float32([array[0] for array in batch])
        
        actions_array = np.float32([array[1] for array in batch])
        rewards_array = np.float32([array[2] for array in batch])
        next_states_array = np.float32([array[3] for array in batch])
        dones = np.bool8([array[4] for array in batch])
        
        return states_array, actions_array, rewards_array, next_states_array, dones
    
    def len(self):
        return self.len
    
    def add(self, s, a, r, new_s, d):
        transition = (s, a, r, new_s, d)
        self.len += 1 
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)


class DuelingQAgent():
    def __init__(self, state_size, action_size, mode, load_eps):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('dueling_dqn_gazebo/nodes', 'dueling_dqn_gazebo/save_model/Dueling_DQN_trial_1/pt_trial_1_')
        self.result = Float32MultiArray()

        self.mode = mode
        self.load_model = False
        self.load_episode = load_eps
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000   #2000    # after target_update's time, Target network will be updated
        self.discount_factor = 0.995
        self.learning_rate = 0.001      #0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.997      #0.996
        self.epsilon_min = 0.01
        self.tau = 0.01
        self.batch_size =64
        self.train_start = 64
        self.memory_size = 45000
        self.RAM = MemoryBuffer(self.memory_size)
        
        self.Pred_model = DuelingQNetwork(self.state_size, self.action_size)
        self.Target_model = DuelingQNetwork(self.state_size, self.action_size)
        
        self.optimizer = optim.AdamW(self.Pred_model.parameters(), self.learning_rate) # Adam Optimizer but with L2 Regularization implemented
        self.loss_func = nn.MSELoss()
        self.episode_loss = 0.0
        self.running_loss = 0.0
        self.training_loss = []
        self.x_episode = []
        self.counter = 0
        
        if self.mode == "test":
            self.load_model = True
        
        # Switches between training on initial weights and weights loaded from the pre-trained episode
        if self.load_model:
            loaded_state_dict = torch.load(self.dirPath+str(self.load_episode)+'.pt')
            self.Pred_model.load_state_dict(loaded_state_dict)

    def updateTargetModel(self):
        self.Target_model.load_state_dict(self.Pred_model.state_dict())

    def getAction(self, state):
        
        if len(state.shape)==2:
            state = torch.from_numpy(state)
            state=state.unsqueeze(0).view(1, 1, 144, 176)
            self.q_value = self.Pred_model(state)
            
            if self.mode == "train":
                if np.random.rand() <= self.epsilon:
                    self.q_value = np.zeros(self.action_size)
                    action = random.randrange(self.action_size)
                else:
                    action = int(torch.argmax(self.q_value))
                    print("(*) Predicted action: ", action)
            if self.mode =="test":
                action = int(torch.argmax(self.q_value))
            return action
        
        
    def TrainModel(self):
        states, actions, rewards, next_states, dones = self.RAM.sample(self.batch_size)
        states = np.array(states).squeeze()
        next_states = np.array(next_states).squeeze()
        states = torch.tensor(states).view(64, 1, 144, 176)
        next_states = torch.tensor(next_states).view(64, 1, 144, 176)

        actions = torch.Tensor(actions)
        actions = actions.type(torch.int64).unsqueeze(-1)
        next_q_value = torch.max(self.Target_model(next_states), dim=1)[0]
        
        ## check if the episode terminates in next step
        q_value = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if dones[i]:
                q_value[i] = rewards[i]
            else:
                q_value[i] = rewards[i] + self.discount_factor * next_q_value[i]

        ## convert td_target to tensor
        td_target = torch.Tensor(q_value)
   
        ## get predicted_values
        predicted_values = self.Pred_model(states).gather(1, actions).squeeze()

      
        ## calculate the loss 
        self.loss = self.loss_func(predicted_values, td_target)
         
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
        
        
        self.episode_loss += predicted_values.shape[0] * self.loss.item()
        self.running_loss += self.loss.item()
        cal_loss = self.episode_loss / len(states)
        self.training_loss.append(cal_loss)
        self.counter += 1
        self.x_episode.append(self.counter)
        np.savetxt(LOSS_DATA_DIR + '/loss.csv', self.training_loss, delimiter = ' , ')
        