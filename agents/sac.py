import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class SAC(nn.Module):

    def __init__(self, state_dim, action_dim, buffer_size, alpha, gamma, tau, lr, device):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.buffer = ReplayBuffer(buffer_size)

        # Actor
        self.actor = Actor(state_dim, action_dim)

        # Q Networks
        self.q1 = Q_Critic(state_dim, action_dim)
        self.q2 = Q_Critic(state_dim, action_dim)

        # Value Network and target
        self.value = V_Critic(state_dim)
        self.value_target = V_Critic(state_dim)
        self.value_target.load_state_dict(self.value.state_dict()) # Copy weights to target

        # Optimizers
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr = lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr = lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr = lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = lr)
   

    def act(self, state, deterministic=False):

        action, _ = self.actor.sample(state, deterministic)
        return action.detach().numpy()
    

    def save_to_buffer(self, transition):

        self.buffer.put(transition)
    

    def update(self, batch_size):

        states, actions, rewards, next_states, terminateds = self.buffer.sample(batch_size)

        states = torch.stack(states) # (batch_size, obs_size)
        next_states = torch.stack(next_states) # (batch_size, obs_size)
        actions = torch.as_tensor(np.array(actions), dtype=torch.float32, device=self.device) # (batch_size, action_size)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)  # (batch_size, 1)
        terminateds = torch.as_tensor(np.array(terminateds), dtype=torch.float32, device=self.device).unsqueeze(1) # (batch_size, 1)

        # Q_network update
        q_target = (rewards + (1 - terminateds) * self.gamma * self.value_target(next_states)).detach()

        q1_value = self.q1(states, actions)
        q2_value = self.q2(states, actions)

        q1_loss = F.mse_loss(q1_value, q_target)
        q2_loss = F.mse_loss(q2_value, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()


        # Value update
        sampled_action, log_prob = self.actor.sample(states)
        q_min = torch.min(self.q1(states, sampled_action), self.q2(states, sampled_action))
        value_target = (q_min - self.alpha * log_prob).detach()

        value_loss = F.mse_loss(self.value(states), value_target)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


        # Policy update
        sampled_action, log_prob = self.actor.sample(states)
        q_min = torch.min(self.q1(states, sampled_action), self.q2(states, sampled_action))
        actor_loss = (self.alpha * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update of Target_V
        for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save_state(self, timestep, avg_returns, filename):
            
        checkpoint = {
            "buffer": self.buffer.buffer,
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "value": self.value.state_dict(),
            "value_target": self.value_target.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "timestep" : timestep,
            "avg_returns" : avg_returns
        }
        torch.save(checkpoint, filename)


    def load_state(self, filename):

        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)

        self.buffer.buffer = checkpoint["buffer"]
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.value.load_state_dict(checkpoint["value"])
        self.value_target.load_state_dict(checkpoint["value_target"])
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        timestep = checkpoint["timestep"]
        avg_returns = checkpoint["avg_returns"]

        return (timestep, avg_returns)


class Actor(nn.Module):
         
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_means = nn.Linear(256, action_dim)
        self.fc_log_std = nn.Linear(256, action_dim)

    def forward(self, state):
                 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_means(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)

        return mean, std
    
    def sample(self, state, deterministic=False):

        mean, std = self.forward(state)
        m = torch.distributions.Normal(mean, std)
        if deterministic:
            z = mean
        else:
            z = m.rsample()

        action = torch.tanh(z)

        # Log probability with Tanh correction
        log_prob = m.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob


class Q_Critic(nn.Module):
         
    def __init__(self, state_dim, action_dim):
        super().__init__()
                
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
                 
        x = torch.cat([state, action], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value
    

class V_Critic(nn.Module):
         
    def __init__(self, state_dim):
        super().__init__()
                
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state):
                 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        return value


class ReplayBuffer():

    def __init__(self, size):
        self.buffer = deque(maxlen = size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done= transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones