import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import pygame
import os
import glob
import imageio
from collections import deque


class FrameStack:
    def __init__(self, num_envs, frame_stack, height=84, width=84, device="cpu"):
        self.num_envs = num_envs
        self.frame_stack = frame_stack
        self.height = height
        self.width = width
        self.device = device

        # Buffer: (num_envs, frame_stack, height, width)
        self.frames = np.zeros((num_envs, frame_stack, height, width), dtype=np.float32)

    def reset(self, first_obs):
        """
        Called after env.reset() -> fill all frames with the first observation.
        first_obs: (num_envs, height, width) already preprocessed (grayscale, 84x84).
        """
        if first_obs.ndim == 3:  # (H, W, C)
            first_obs = np.expand_dims(first_obs, axis=0) 

        phi_frames = self.preprocess_frames(first_obs)
        for i in range(self.frame_stack):
            self.frames[:, i] = phi_frames
        return torch.tensor(self.frames, device=self.device)

    def step(self, new_frames):
        """
        Called after env.step() -> preprocess and add new obs and shift frames.
        new_obs: (num_envs, height, width, 3) raw frames.
        """
        if new_frames.ndim == 3:  # (H, W, C)
            new_frames = np.expand_dims(new_frames, axis=0) 

        phi_frames = self.preprocess_frames(new_frames)
        self.frames[:, :-1] = self.frames[:, 1:]  # shift left
        self.frames[:, -1] = phi_frames              # insert new frame
        return torch.tensor(self.frames, device=self.device)
    
    def preprocess_frames(self, new_frames):

        n_env = new_frames.shape[0]
        phi_frames = np.zeros((n_env, 84, 84), dtype=np.float32)

        for i in range(n_env):
            gray = cv.cvtColor(new_frames[i], cv.COLOR_RGB2GRAY)
            resized = cv.resize(gray, (84, 84), interpolation=cv.INTER_AREA)
            phi_frames[i] = resized.astype(np.float32) / 255.0

        return phi_frames


def eval_model(model, env_name, n_episodes_eval, device):

    model.eval()
    env = gym.make(env_name)
    returns = [run_eval_episode(env, model, device) for _ in range(n_episodes_eval)]
    env.close()

    model.train()

    return np.mean(returns)


def run_eval_episode(env, model, device):

    tot_reward = 0
    done = False
    framestack = FrameStack(1, 4, 84, 84, device)

    obs, infos = env.reset()
    state = framestack.reset(obs)
    current_lives = infos['lives'] + 1

    with torch.no_grad():
        while not done:

            actor_logits, value = model(state)
            action = actor_logits.argmax(dim=-1).item()

            if infos['lives'] < current_lives:
                current_lives = infos['lives']
                action = 1

            obs, reward, terminated, truncated, infos = env.step(action)
            done = terminated or truncated
            tot_reward += reward
            state = framestack.step(obs)
    
    return tot_reward
    

def generate_video(model, env_name, filename, device):

    model.eval()
    env = gym.make(env_name)
    frames = []
    
    with torch.no_grad():
        
        framestack = FrameStack(1, 4, 84, 84, device)

        obs, infos = env.reset()
        state = framestack.reset(obs)
        current_lives = infos['lives'] + 1
        done = False

        while not done:

            actor_logits, value = model(state)
            action = actor_logits.argmax(dim=-1).item()

            if infos['lives'] < current_lives:
                current_lives = infos['lives']
                action = 1

            obs, reward, terminated, truncated, infos = env.step(action)
            frames.append(cv.resize(obs, (160, 224)))
            done = terminated or truncated
            state = framestack.step(obs)

    
    imageio.mimsave(filename, frames, fps=30)

    env.close()
    model.train()


def save_checkpoint(model, optimizer, timestep, losses, avg_returns, filename="checkpoint.pth"):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "timestep" : timestep,
        "losses" : losses,
        "avg_returns" : avg_returns
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename="checkpoint.pth", device="cpu"):
    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    timestep = checkpoint['timestep']
    losses = checkpoint['losses']
    avg_returns = checkpoint['avg_returns']

    return  timestep, losses, avg_returns


def save_plots(losses, avg_returns, path, timestep, eval_frequency, plot_losses=True):

    for file in glob.glob(f"{path}/*.png"): os.remove(file)

    # Optionally apply a simple moving average for smoother curves
    def moving_average(x, window=50):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    if plot_losses:

        losses, actor_losses, critic_losses, entropies = tuple(zip(*losses))

        losses = np.array(losses)
        actor_losses = np.array(actor_losses)
        critic_losses = np.array(critic_losses)
        entropies = np.array(entropies)

        # Plot
        plt.figure(figsize=(14, 8))

        # Total loss
        plt.subplot(2, 2, 1)
        plt.plot(losses, label="Total Loss", alpha=0.7)
        plt.plot(moving_average(losses), label="Smoothed", linewidth=2)
        plt.xlabel("Updates")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Total Loss")

        # Actor loss
        plt.subplot(2, 2, 2)
        plt.plot(actor_losses, label="Actor Loss", alpha=0.7)
        plt.plot(moving_average(actor_losses), label="Smoothed", linewidth=2)
        plt.xlabel("Updates")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Actor Loss")

        # Critic loss
        plt.subplot(2, 2, 3)
        plt.plot(critic_losses, label="Critic Loss", alpha=0.7)
        plt.plot(moving_average(critic_losses), label="Smoothed", linewidth=2)
        plt.xlabel("Updates")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Critic Loss")

        # Entropy
        plt.subplot(2, 2, 4)
        plt.plot(entropies, label="Entropy", alpha=0.7)
        plt.plot(moving_average(entropies), label="Smoothed", linewidth=2)
        plt.xlabel("Updates")
        plt.ylabel("Entropy")
        plt.legend()
        plt.title("Entropy")

        plt.tight_layout()

        plt.savefig(f"{path}/losses_{timestep}.png", dpi=300, bbox_inches="tight")
        plt.close()


    plt.figure(figsize=(8, 8))
    timesteps = np.arange(1, len(avg_returns)+1) * eval_frequency
    smoothed = moving_average(avg_returns, window=10)
    smoothed_timesteps = timesteps[len(timesteps) - len(smoothed):]
    plt.plot(timesteps, avg_returns, label="Average return", alpha=0.7)
    plt.plot(smoothed_timesteps, smoothed, label="Smoothed", linewidth=2)
    plt.xlabel("Timesteps")
    plt.ylabel("Average return")
    plt.legend()
    plt.title("Average return per episode")

    plt.savefig(f"{path}/return_{timestep}.png", dpi=300, bbox_inches="tight")
    plt.close()


def play_manual_breakout(save_probability=.1):
    """
    Play Breakout manually using keyboard (hold LEFT/RIGHT to keep moving).
    Collect sampled states into `collected_states`.
    """

    # Create environment — set frameskip=1 for responsive manual control
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    # Try to get action meanings (helps map keys to correct action indices)
    try:
        action_meanings = env.get_action_meanings()
    except Exception:
        try:
            action_meanings = env.unwrapped.get_action_meanings()
        except Exception:
            action_meanings = None

    print("Action meanings:", action_meanings)

    # Build an action mapping from names -> indices (best-effort)
    action_map = {}
    if action_meanings:
        for i, name in enumerate(action_meanings):
            name = name.upper()
            if "NOOP" in name:
                action_map["noop"] = i
            if "FIRE" == name:
                action_map["fire"] = i
            if "RIGHT" in name:
                # RIGHT or RIGHTFIRE
                # prefer pure RIGHT if present, else the first RIGHT* variant
                action_map.setdefault("right", i)
            if "LEFT" in name:
                action_map.setdefault("left", i)

    # Fallback defaults if detection failed
    action_map.setdefault("noop", 0)
    action_map.setdefault("fire", 1)
    action_map.setdefault("right", 2)
    action_map.setdefault("left", 3)

    print("Using action_map:", action_map)

    # Initialize pygame to capture keyboard (small control window)
    pygame.init()
    control_w, control_h = 360, 80
    screen = pygame.display.set_mode((control_w, control_h))
    clock = pygame.time.Clock()

    try:
        frame, info = env.reset()
        phi_frame = preprocess_frame(frame)
        last_frames = deque([phi_frame] * 4, maxlen=4)  # fill initial stack
        state = get_state(last_frames)

        done = False
        truncated = False
        collected_states = []   
        running = True

        while running:
            # Process pygame events (handle quit)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # optional: allow ESC to quit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            # Poll keyboard state so holding the key works
            keys = pygame.key.get_pressed()

            # Default action = NOOP
            action = action_map["noop"]

            # Prioritize movement keys: left/right -> then fire
            if keys[pygame.K_LEFT]:
                action = action_map["left"]
            elif keys[pygame.K_RIGHT]:
                action = action_map["right"]
            elif keys[pygame.K_SPACE]:
                action = action_map["fire"]

            # Step environment with the chosen action
            frame, reward, done, truncated, info = env.step(action)
            phi_frame = preprocess_frame(frame)
            last_frames.append(phi_frame)
            state = get_state(last_frames)  # shape (4,84,84)

            # Optionally save the state stack
            if np.random.random() < save_probability:
                collected_states.append(state)

            pygame.display.flip()

            # Stop if env ended
            if done or truncated:
                running = False

            # Run the loop at 60Hz (good responsiveness)
            clock.tick(15)

    # Close env and pygame
    finally:
        env.close()
        pygame.quit()

    return collected_states


def preprocess_frame(frame):
    
    phi_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    phi_frame = cv.resize(phi_frame, (84, 84))

    return phi_frame


def get_state(last_frames, device):

    state = np.stack(last_frames, axis=0)  # shape: (4, 84, 84)
    state = state.astype(np.float32)
    state /= 255.0 # Normalize to [0,1]
    state = torch.from_numpy(state).unsqueeze(0).to(device) # shape : (1, 4, 84, 84)
    
    return state


def display_state(state):
    fig, ax = plt.subplots(1, 4)
    for i in range(4):
        ax[i].imshow(state[0,i])
