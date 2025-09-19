import gymnasium as gym
import panda_gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import glob
import os

def eval_model(model, env_name, n_episodes_eval):

    env = gym.make(env_name)
    returns = [run_eval_episode(env, model) for _ in range(n_episodes_eval)]
    env.close()

    return np.mean(returns)


def run_eval_episode(env, model):

    obs, info = env.reset()
    state = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]]))
    done = False
    tot_reward = 0

    with torch.no_grad():

        while not done:

            action = model.act(state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            state = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]]))
            done = terminated or truncated
            tot_reward += reward
    
    return tot_reward

def generate_video(env_name, model, n_episodes, deterministic, filename):

    env = gym.make(env_name, render_mode="rgb_array")
    frames = []

    with torch.no_grad():

        for i in range(n_episodes):

            obs, info = env.reset()
            state = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]]))
            done = False

            while not done:

                action = model.act(state, deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                state = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]]))
                done = terminated or truncated

                frames.append(env.render())

    imageio.mimsave(filename, frames, fps=30)
    env.close()

def save_plots(avg_returns, path, timestep, eval_frequency):

    for file in glob.glob(f"{path}/*.png"): os.remove(file)

    def moving_average(x, window=10):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode='valid')
    

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