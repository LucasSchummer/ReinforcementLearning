import gymnasium as gym
import panda_gym
from panda_gym.utils import distance
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import glob
import os


def compute_reward(achieved_goal, desired_goal, distance_threshold):

    d = distance(achieved_goal, desired_goal)
    return -np.array(d > distance_threshold, dtype=np.float32)

def eval_model(model, env_name, max_episode_steps, n_episodes_eval):

    env = gym.make(env_name, max_episode_steps=max_episode_steps)
    returns, successes = [], []
    for _ in range(n_episodes_eval):
        ep_return, success = run_eval_episode(env, model)
        returns.append(ep_return)
        successes.append(success)

    env.close()

    return np.mean(returns), np.mean(successes)


def run_eval_episode(env, model):

    obs, info = env.reset()
    while info.get("is_success", False): # Reset env if start is success state
        obs, info = env.reset()

    state_goal = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]]))
    done = False
    tot_reward = 0

    with torch.no_grad():

        while not done:

            action = model.act(state_goal, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            state_goal = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]]))
            done = terminated or truncated
            tot_reward += reward
    
    return tot_reward, info.get("is_success", False)

def generate_video(env_name, max_episode_steps, model, n_episodes, random, deterministic, filename):

    env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    frames = []

    with torch.no_grad():

        for i in range(n_episodes):

            obs, info = env.reset()
            state_goal = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]]))
            done = False

            while not done:

                if not random:
                    action = model.act(state_goal, deterministic)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                state_goal = torch.tensor(np.concatenate([obs["observation"], obs["desired_goal"]]))
                done = terminated or truncated

                frames.append(env.render())

    imageio.mimsave(filename, frames, fps=30)
    env.close()

def save_plots(avg_returns, avg_successes, path, timestep, eval_frequency):

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


    plt.figure(figsize=(8, 8))
    smoothed = moving_average(avg_successes, window=10)
    smoothed_timesteps = timesteps[len(timesteps) - len(smoothed):]
    plt.plot(timesteps, avg_successes, label="Success rate", alpha=0.7)
    plt.plot(smoothed_timesteps, smoothed, label="Smoothed", linewidth=2)
    plt.xlabel("Timesteps")
    plt.ylabel("Success rate")
    plt.legend()
    plt.title("Eval success rate")

    plt.savefig(f"{path}/success_{timestep}.png", dpi=300, bbox_inches="tight")
    plt.close()