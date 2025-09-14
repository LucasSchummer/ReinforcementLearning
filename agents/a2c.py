import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class A2C(nn.Module):
    def __init__(self, input_channels, n_actions = 4, gamma = .99, max_grad_norm = .5, c_actor = 1, c_critic = .25, c_entropy = .01, device = "cpu"):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1)

        self.fc = nn.Linear(64 * 7 * 7, 512)

        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.c_actor = c_actor
        self.c_critic = c_critic
        self.c_entropy = c_entropy
        self.device = device
            
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))

        actor_logits = self.actor(x)
        value = self.critic(x)

        return actor_logits, value
      
    def update(self, optimizer, logits_buffer, log_probs_buffer, values_buffer, rewards_buffer, next_values_buffer, dones_buffer):
            
        logits = torch.stack(logits_buffer) # (batch_size, n_env, n_actions)
        log_probs = torch.stack(log_probs_buffer) # (batch_size, n_env)
        values = torch.stack(values_buffer) # (batch_size, n_env)
        next_values = torch.stack(next_values_buffer) # (batch_size, n_env)
        rewards = torch.as_tensor(np.array(rewards_buffer), dtype=torch.float32, device=self.device)  # (batch_size, n_env)
        dones = torch.as_tensor(np.array(dones_buffer), dtype=torch.float32, device=self.device) # (batch_size, n_env)

        targets = rewards + (1 - dones) * self.gamma * next_values
        advantages = targets - values

        B = advantages.shape[0] * advantages.shape[1]  # total batch size
        advantages = advantages.view(B)
        values = values.view(B)
        targets = targets.view(B)
        log_probs = log_probs.view(B)
        logits = logits.view(B, -1) 

        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()

        critic_loss = 0.5 * advantages.pow(2).mean()
        actor_loss = -(log_probs * advantages.detach()).mean()

        loss = self.c_actor * actor_loss + self.c_critic * critic_loss - self.c_entropy * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

        return (loss.item(), actor_loss.item(), critic_loss.item(), entropy.item())

