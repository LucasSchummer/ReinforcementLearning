import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from utils import *

class QNetwork(nn.Module):
      def __init__(self, input_dim, hidden_dim = 256, output_dim = 3):
            super().__init__()
            self.fc   = nn.Linear(input_dim, hidden_dim)
            self.fcQ1 = nn.Linear(hidden_dim, hidden_dim)
            self.fcQ2 = nn.Linear(hidden_dim, output_dim)

      def forward(self, states):
            states  = F.relu(self.fc(states))
            states  = F.relu(self.fcQ1(states))
            actions = self.fcQ2(states)
            return actions

class ReplayBuffer():
      def __init__(self):
            self.buffer = deque(maxlen = 100000)

      def put(self, transition):
            self.buffer.append(transition)

      def sample(self, n):
            mini_batch = random.sample(self.buffer, n)
            states, actions, rewards, next_states, terminateds, truncateds = [], [], [], [], [], []

            for transition in mini_batch:
                  state, action, reward, next_state, terminated, truncated = transition
                  states.append(state)
                  actions.append(action)
                  rewards.append(reward)
                  next_states.append(next_state)
                  terminateds.append(terminated)
                  truncateds.append(truncated)

            return states, actions, rewards, next_states, terminateds, truncateds

      def size(self):
            return len(self.buffer)

def Update_Q(buffer, Q, Q_target, Q_optimizer, batch_size, gamma):
      states, actions, rewards, next_states, terminateds, truncateds = buffer.sample(batch_size)

      states      = torch.stack(states)                        # (batch_size, state_dim)
      next_states = torch.stack(next_states)                   # (batch_size, state_dim)
      actions     = torch.tensor(actions, dtype=torch.long)    # (batch_size,)
      rewards     = torch.tensor(rewards, dtype=torch.float32) # (batch_size,)
      dones       = torch.tensor(
            [t or tr for t, tr in zip(terminateds, truncateds)],
            dtype=torch.float32
      )  

      with torch.no_grad():
            target_q_values = Q_target(next_states)              # (batch_size, n_actions)
            max_next_q = target_q_values.max(dim=1)[0]           # (batch_size,)
            y = rewards + gamma * (1 - dones) * max_next_q       # (batch_size,)

      # Compute current Q-values for chosen actions
      q_values = Q(states)                                     # (batch_size, n_actions)
      q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch_size,)

      # Compute loss
      loss = F.mse_loss(q_value, y)

      Q_optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(Q.parameters(), 10.0)
      Q_optimizer.step()

      return loss.item()

def save_checkpoint(q_net, optimizer, replay_buffer, returns, avg_Qvalues, td_losses, episode, epsilon, filename="checkpoint.pth"):
    checkpoint = {
        "q_network": q_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay_buffer": replay_buffer.buffer,  # deque
        "returns" : returns,
        "avg_Qvalues" : avg_Qvalues,
        "td_losses" : td_losses,
        "episode": episode,
        "epsilon" : epsilon
    }
    torch.save(checkpoint, filename)
    # print(f"Checkpoint saved to {filename}")

def load_checkpoint(q_net, optimizer, replay_buffer, filename="checkpoint.pth", device="cpu"):
    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    q_net.load_state_dict(checkpoint["q_network"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Restore replay buffer
    replay_buffer.buffer = checkpoint["replay_buffer"]

    returns = checkpoint['returns']
    avg_Qvalues = checkpoint['avg_Qvalues']
    td_losses = checkpoint['td_losses']
    episode = checkpoint["episode"] + 1
    epsilon = checkpoint["epsilon"]

    # print(f"Checkpoint loaded from {filename}, resuming at episode {episode}")
    return returns, avg_Qvalues, td_losses, episode, epsilon

def get_avg_Qvalues(Q, states):

    return np.mean([torch.mean(Q(state)).item() for state in states])


def evaluate_average_return(Q, env, device, n_episodes, action_map):

    returns = []
    victories = 0
    for episode in range(n_episodes):

        total_reward = 0

        # Skip first frame (different color)
        frame, _ = env.reset()
        _ = env.step(0)

        frame, reward, terminated, truncated, info = env.step(0)
        ball_position, player_position, state = get_state(frame, (0, np.array([0, 0])))
        state = torch.tensor(state).float().to(device)

        done = False
        while not done:

            # greedy action selection
            action = torch.argmax(Q(state)).item()

            frame, reward, terminated, truncated, info = env.step(action_map[action])

            ball_position, player_position, state = get_state(frame, ball_position, player_position)
            state = torch.tensor(state).float().to(device)

            total_reward += reward

            done = terminated or truncated

        returns.append(total_reward)
        if reward == 1 : victories += 1

    return np.mean(returns), victories / n_episodes