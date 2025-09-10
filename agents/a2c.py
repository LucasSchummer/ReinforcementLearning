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
      

def update_network(model, optimizer, device, logits, log_probs, values, rewards, next_values, dones, gamma, c_actor, c_critic, c_entropy, max_grad_norm):

    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    values = torch.stack(values).to(device)
    next_values = torch.stack(next_values).to(device)
    log_probs = torch.stack(log_probs).to(device)
    logits_tensor = torch.stack(logits).to(device)

    target = rewards + (1 - dones) * gamma * next_values
    advantages = target - values
 
    entropy = torch.distributions.Categorical(logits=logits_tensor).entropy().mean()

    critic_loss = 0.5 * advantages.pow(2).mean()
    actor_loss = - (log_probs * advantages.detach()).mean() # No grad through advantage

    loss = c_critic * critic_loss + c_actor * actor_loss - c_entropy * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
