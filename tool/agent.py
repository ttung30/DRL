#!usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import rospy
import random
import numpy as np
from collections import deque
from std_msgs.msg import Float32MultiArray
from tool.model import DuelingQNetwork


LOSS_DATA_DIR = '/log/Dueling_DQN_trial_1/log_data'


class MemoryBuffer():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        

    def sample(self, count):
        """_summary_

        Args:
            count (_type_): _description_

        Returns:
            _type_: _description_
        """
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
        self.dirPath = 'log/Dueling_DQN_trial_1/pt_trial_1_'
        self.result = Float32MultiArray()
        self.taus= torch.linspace(0.0, 1.0, 51, dtype=torch.float32)
        self.mode = mode
        self.load_model = False
        self.load_episode = load_eps
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000   
        self.discount_factor = 0.995
        self.learning_rate = 0.001      
        self.epsilon = 1.0
        self.epsilon_decay = 0.997   
        self.epsilon_min = 0.01
        self.tau = 0.01
        self.batch_size =64
        self.train_start = 64
        self.memory_size = 45000
        self.RAM = MemoryBuffer(self.memory_size)
        self.Pred_model = DuelingQNetwork()
        self.Target_model = DuelingQNetwork()
        self.optimizer = optim.AdamW(self.Pred_model.parameters(), self.learning_rate) # Adam Optimizer but with L2 Regularization implemented
        self.loss_func = nn.MSELoss()
        self.episode_loss = 0.0
        self.running_loss = 0.0
        self.training_loss = []
        self.x_episode = []
        self.counter = 0
        
        if self.mode == "test":
            self.load_model = True
        
        
        if self.load_model:
            loaded_state_dict = torch.load(self.dirPath+str(self.load_episode)+'.pt')
            self.Pred_model.load_state_dict(loaded_state_dict)


    def updateTargetModel(self):
        self.Target_model.load_state_dict(self.Pred_model.state_dict())


    def wasserstein_distance(self,target_distribution, predicted_distribution):
        target_distribution, _ = torch.sort(target_distribution, dim=-1)
        predicted_distribution, _ = torch.sort(predicted_distribution, dim=-1)
        wasserstein_distance = torch.abs(target_distribution - predicted_distribution).sum(dim=-1).mean(dim=-1)
        return wasserstein_distance
    

    def getAction(self, state):
        """_summary_

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(state.shape)==2:
            state = torch.from_numpy(state)
            state=state.unsqueeze(0).view(1, 1, 144, 176)
            self.q_value =torch.mean(self.Pred_model(state), dim=2)
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
        

    def quantile_huber_loss(self,target_quantiles, predicted_quantiles, kappa=1.0):
        errors = target_quantiles- predicted_quantiles
        huber_loss = torch.where(torch.abs(errors) < kappa,
                                0.5 * errors.pow(2),
                                kappa * (torch.abs(errors) - 0.5 * kappa))
        tau_scaled = torch.abs(self.taus.unsqueeze(1).unsqueeze(1) - (errors.detach() < 0).float())
        quantile_loss = tau_scaled * huber_loss
        wasserstein_distance_loss =self.wasserstein_distance(target_quantiles, predicted_quantiles)
        loss = quantile_loss.sum(dim=2).mean(dim=1).mean()
        loss = loss+wasserstein_distance_loss
        return loss


    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(1e-2*local_param.data + (1.0-1e-2)*target_param.data)


    def TrainModel(self):
        """_summary_
        """
        states, actions, rewards, next_states, dones = self.RAM.sample(self.batch_size)
        states = np.array(states).squeeze()
        next_states = np.array(next_states).squeeze()
        states = torch.tensor(states).view(64, 1, 144, 176)
        next_states = torch.tensor(next_states).view(64, 1, 144, 176)
        actions = torch.Tensor(actions)
        actions = actions.type(torch.int64).unsqueeze(-1)
        next_q_value = torch.max(self.Target_model(next_states), dim=1)[0].detach().numpy()
        q_value = np.zeros((self.batch_size,51))
        for i in range(self.batch_size):
            if dones[i]:
                q_value[i] = rewards[i]
            else:
                q_value[i] = rewards[i]+self.discount_factor * next_q_value[i]
        td_target = torch.Tensor(q_value)
        tung=self.Pred_model(states)
        predicted_values = tung.gather(1, actions.view(64, 1, 1).expand(64, 1, tung.size(2))).squeeze()
        self.loss=self.quantile_huber_loss(predicted_values, td_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.soft_update(self.Pred_model, self.Target_model)
        self.episode_loss += predicted_values.shape[0] * self.loss.item()
        self.running_loss += self.loss.item()
        cal_loss = self.episode_loss / len(states)
        self.training_loss.append(cal_loss)
        self.counter += 1
        self.x_episode.append(self.counter)
        np.savetxt(LOSS_DATA_DIR + '/loss.csv', self.training_loss, delimiter = ' , ')
        