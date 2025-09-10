import torch
import torch.nn as nn
import torch.nn.functional as F


class A2C(nn.Module):
      def __init__(self, input_channels, n_actions = 4):
            super().__init__()

            self.conv1 = nn.Conv2d(input_channels, out_channels=32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1)

            self.fc = nn.Linear(64 * 7 * 7, 512)

            self.actor = nn.Linear(512, n_actions)
            self.critic = nn.Linear(512, 1)
            
      def forward(self, x):

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

            x = torch.flatten(x, 1)
            x = F.relu(self.fc(x))

            actor_logits = self.actor(x)
            value = self.critic(x)

            return actor_logits, value
      

def update_network(model, optimizer, logits, log_probs, values, rewards, next_values, dones, gamma, c_actor, c_critic, c_entropy, max_grad_norm):

    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    target = rewards + (1 - dones) * gamma * torch.stack(next_values)
    advantages = target - torch.stack(values)
 
    logits_tensor = torch.stack(logits) 
    entropies = [torch.distributions.Categorical(logits=l).entropy() for l in logits_tensor]
    entropy = torch.stack(entropies).mean()

    critic_loss = 0.5 * advantages.pow(2).mean()
    actor_loss = - (torch.stack(log_probs) * advantages.detach()).mean() # No grad through advantage

    loss = c_critic * critic_loss + c_actor * actor_loss - c_entropy * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
