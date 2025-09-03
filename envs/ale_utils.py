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


def preprocess_frame(frame):
    
    phi_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    phi_frame = cv.resize(phi_frame, (84, 84))

    return phi_frame

def get_state(last_frames):

    state = np.stack(last_frames, axis=0)  # shape: (4, 84, 84)
    state = state.astype(np.float32)
    state /= 255.0 # Normalize to [0,1]
    state = torch.from_numpy(state).unsqueeze(0).to("cpu") # shape : (1, 4, 84, 84)
    
    return state


def display_state(state):
    fig, ax = plt.subplots(1, 4)
    for i in range(4):
        ax[i].imshow(state[0,i])


def setup_training_dir(resume_training, algo, version):
    training_numbers = [int(folder.split("training")[-1]) for folder in glob.glob(f"training/a2c/{version}/*")]
    if resume_training:
        training_number = max(training_numbers)
    else:
        training_number = max(training_numbers) + 1 if len(training_numbers) > 0 else 1
    os.makedirs(f"training/{algo}/{version}/training{training_number}", exist_ok=True)
    return training_number


def save_checkpoint(model, optimizer, returns, avg_values, episode, filename="checkpoint.pth"):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "returns" : returns,
        "avg_values" : avg_values,
        "episode": episode,
    }
    torch.save(checkpoint, filename)
    # print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint.pth", device="cpu"):
    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    returns = checkpoint['returns']
    avg_values = checkpoint['avg_values']
    episode = checkpoint["episode"] + 1

    # print(f"Checkpoint loaded from {filename}, resuming at episode {episode}")
    return returns, avg_values, episode


def generate_video(env, model, frame_stack, n_episodes, filename):

    frames = []
    
    for episode in range(n_episodes):

        last_frames = deque(maxlen=frame_stack)

        frame, _ = env.reset()
        phi_frame = preprocess_frame(frame)

        # Initially, fill the last_frames buffer with the first frame
        for _ in range(frame_stack):
            last_frames.append(phi_frame)

        state = get_state(last_frames)

        done = False
        while not done:

            actor_logits, value = model(state)
            m = torch.distributions.Categorical(logits=actor_logits)
            action = m.sample()

            frame, reward, done, truncated, info = env.step(action.item())
            frames.append(cv.resize(frame, (160, 224)))
            phi_frame = preprocess_frame(frame)

            last_frames.append(phi_frame) # Automatically removes the oldest frame

    
    imageio.mimsave(filename, frames, fps=30)


def save_plots(returns, avg_values, episode, path):

    for file in glob.glob(f"{path}/*.png"): os.remove(file)
    
    fig1, ax1 = plt.subplots()
    avg_returns = [np.mean(returns[i-100:i]) for i in range(100, len(returns))]
    ax1.plot(range(100, len(returns)), avg_returns)
    ax1.set_title("Average return per episode (100 last episodes)")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Average Return")

    
    fig2, ax2 = plt.subplots()
    moving_avg_values = [np.mean(avg_values[i-100:i]) for i in range(100, len(avg_values))]
    ax2.plot(range(100, len(avg_values)), moving_avg_values)
    ax2.set_title("Average value (moving average on last 100 episodes)")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Average value")

    fig1.savefig(f"{path}/return_{episode}.png")
    fig2.savefig(f"{path}/value_{episode}.png")
    plt.close(fig1)
    plt.close(fig2)


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