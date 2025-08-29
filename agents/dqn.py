import random
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from pong_utils import *



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



# # --- Pygame init ---
# pygame.init()
# screen = pygame.display.set_mode((400, 300))
# pygame.display.set_caption("Play Pong with Keyboard")
# clock = pygame.time.Clock()

# # --- Gym init ---
# env = gym.make("ALE/Pong-v5", render_mode="human")
# obs, info = env.reset()

# # Mapping: keys -> actions
# # 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4: RIGHTFIRE, 5: LEFTFIRE
# key_action_map = {
#     pygame.K_UP: 2,     # Move up (RIGHT in Pong's terms)
#     pygame.K_DOWN: 3,   # Move down (LEFT in Pong's terms)
#     pygame.K_SPACE: 1,  # Fire (start the game)
# }

# done = False
# while True:
#     action = 0  # Default NOOP

#     # --- Handle events ---
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             env.close()
#             pygame.quit()
#             raise SystemExit
        
#         if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
#             env.close()
#             pygame.quit()
#             raise SystemExit

#     # Get pressed keys
#     keys = pygame.key.get_pressed()
#     for key, mapped_action in key_action_map.items():
#         if keys[key]:
#             action = mapped_action

#     # Step the environment
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated

#     if done:
#         obs, info = env.reset()

#     clock.tick(60)  # Limit loop to 60 FPS