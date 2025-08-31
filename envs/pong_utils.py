import random
import torch
import imageio
import numpy as np
import cv2 as cv


BALL_COLOR = (236, 236, 236)
PLAYER_COLOR = (92, 186, 92)
ADVERSARY_COLOR = (213, 130, 74)

action_map = {
    0 : 0, # NOOP
    1 : 2, # RIGHT
    2 : 3 # LEFT
}


def get_state(new_frame, old_ball_position, old_player_position=-1):

    def find_ball(frame, color):
        
        play_space = frame[34:194]
        mask = np.all(play_space == color, axis=-1)

        # Check if the ball is visible and return its position
        if np.sum(mask) > 0:
            return 1, np.mean(np.argwhere(mask), axis=0)
        else:
            return 0, np.array([0,0])
        
    def find_paddle(frame, color):
        
        play_space = frame[34:194]
        mask = np.all(play_space == color, axis=-1)

        # Check if the ball is visible and return its position
        if np.sum(mask) > 0:
            return 1, np.mean(np.argwhere(mask), axis=0)[0]
        else:
            return 0, 0

    _, player_coord = find_paddle(new_frame, PLAYER_COLOR) # Player is always visible

    # Compute the player speed if the old position is available
    if old_player_position != -1:
        player_speed = player_coord - old_player_position
    else:
        player_speed = 0

    visible_ball, ball_coord = find_ball(new_frame, BALL_COLOR) # Try to find the ball on the current frame
    visible_adv, adv_coord = find_paddle(new_frame, ADVERSARY_COLOR) # Try to find the ball on the current frame

    old_visible_ball, old_ball_coord = old_ball_position

    # Compute the speed of the ball if possible
    if visible_ball and old_visible_ball:
        known_ball_speed = 1
        ball_speed = ball_coord - old_ball_coord
    else :
        known_ball_speed = 0
        ball_speed = np.array([0, 0])


    ball_position = (visible_ball, ball_coord)
    state = np.concatenate([[visible_ball, known_ball_speed, visible_adv], ball_coord, ball_speed, [adv_coord, player_coord, player_speed]])

    return ball_position, player_coord, state


def generate_evaluation_states(env, device, n_states):
    
    states = []

    # Sample one state per episode generated
    for episode in range(n_states):

        # Skip first frame (different color)
        frame, _ = env.reset()
        _ = env.step(0)

        frame, reward, terminated, truncated, info = env.step(0)
        ball_position, player_position, state = get_state(frame, (0, np.array([0, 0])))
        state = torch.tensor(state).float().to(device)

        waiting_frames = np.random.randint(0, 200)

        for t in range(waiting_frames):

            # Perform given number of random actions
            frame, reward, terminated, truncated, info = env.step(env.action_space.sample())
            ball_position, player_position, state = get_state(frame, ball_position, player_position)
            state = torch.tensor(state).float().to(device)

            if terminated or truncated:
                    break
            

        states.append(state.clone())

    return states


def get_avg_Qvalues(Q, states):

    return np.mean([torch.mean(Q(state)).item() for state in states])


def evaluate_average_return(Q, env, device, n_episodes):

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


def generate_video(env, Q, device, epsilon, n_episodes, filename):

    frames = []
    
    for episode in range(n_episodes):

        # Skip first frame (different color)
        frame, _ = env.reset()
        _ = env.step(0)

        frame, reward, terminated, truncated, info = env.step(0)
        ball_position, player_position, state = get_state(frame, (0, np.array([0, 0])))
        state = torch.tensor(state).float().to(device)

        done = False
        while not done:

            # epsilon-greedy action
            if random.random() < epsilon:
                action = np.random.choice(list(action_map.keys()))
            else:
                action = torch.argmax(Q(state)).item()

            frame, reward, terminated, truncated, info = env.step(action_map[action])
            frames.append(cv.resize(frame, (160, 224)))

            ball_position, player_position, state = get_state(frame, ball_position, player_position)
            state = torch.tensor(state).float().to(device)

            done = terminated or truncated
    
    imageio.mimsave(filename, frames, fps=30)