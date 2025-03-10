import numpy as np
import gymnasium as gym
from IPython.display import clear_output
import time

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        time.sleep(0.5)  # Half second between frames

def run_episode(env, Qtable, render=True):
    frames = []
    state, info = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Qtable[state][:])
        new_state, reward, terminated, truncated, info = env.step(action)

        if render:
            frame = env.render()
            frames.append({
                'frame': frame,
                'state': state,
                'action': action,
                'reward': reward
            })

        state = new_state

    return frames

# Create environment
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human")

Qtable_frozenlake = np.load('frozen_lake_qtable.npy')
# Use your trained Q-table here
frames = run_episode(env, Qtable_frozenlake)
print_frames(frames)

# Don't forget to close the environment
env.close()