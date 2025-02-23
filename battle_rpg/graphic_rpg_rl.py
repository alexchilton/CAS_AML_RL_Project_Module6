
import sys
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import pygame
import torch.nn as nn
import torch.nn.functional as F
import torch
import gymnasium as gym

from graphic_env import BattleEnv
#from graphic_visualizer import GameVisualizer  
from metrics_plotter import plot_training_metrics


def train_agent(total_timesteps = 1000, agent_strength = 10, bandit_strength = 6):
    """
    Train the PPO agent.
    
    Args:
        total_timesteps (int): Number of timesteps to train for
        strength (int): Initial strength parameter for the environment
    """    
    # initialize the enviroment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
    
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
            self.layers = nn.Sequential(
                nn.Linear(channels, channels),
                self.norm1,
                nn.ReLU(),
                nn.Linear(channels, channels),
                self.norm2
            )
        
        def forward(self, x):
            return F.relu(x + self.layers(x))

    class CustomResNetwork(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
            super().__init__(observation_space, features_dim)
            
            n_input = int(np.prod(observation_space.shape))
            
            self.input_norm = nn.LayerNorm(n_input)  # Normalize inputs
            self.input_layer = nn.Sequential(
                nn.Linear(n_input, 128),
                nn.ReLU()
            )
            
            self.res_blocks = nn.ModuleList([
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128)
            ])
            
            self.output_layer = nn.Linear(128, features_dim)

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            x = self.input_norm(observations)  # Apply input normalization
            x = self.input_layer(x)
            for res_block in self.res_blocks:
                x = res_block(x)
            return self.output_layer(x)

    # Modified policy kwargs for PPO initialization
    policy_kwargs = dict(
        features_extractor_class=CustomResNetwork,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]
    )
       
    # initialize the PPO model 
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device='cpu',
        learning_rate=5e-5, 
        n_steps=1024, 
        batch_size=256, 
        n_epochs=40, 
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2, 
        normalize_advantage=True, 
        max_grad_norm=0.5,
        vf_coef = 0.5, # increased from 0.3
        policy_kwargs=policy_kwargs
        #tensorboard_log="./ppo_battle_tensorboard/"
    )

    # to output file
    original_stdout=sys.stdout
    try:
        with open('training_log.txt', 'w') as f:
            sys.stdout=f
            model.learn(
                total_timesteps=total_timesteps,
                progress_bar= True, 
                log_interval= 10)
    finally:
        sys.stdout = original_stdout

    # plot metrics
    with open('training_log.txt', 'r') as f: 
        log_content=f.read()
    plot_training_metrics(log_content)
    
    # Save the model
    model.save("graphic_rpg_model")
    print(f"model saved as 'graphic_rpg_model' " )


def test_agent(num_episodes=5, agent_strength=10, bandit_strength=6):
    """
    Test the trained agent.
    
    Args:
        num_episodes (int): Number of episodes to test
        agent_strength (int): Strength parameter for the agent
        bandit_strength (int): Strength parameter for the bandits
    """
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
    model = PPO.load("graphic_rpg_model")
    wins = 0
    losses = 0 
    
    try:
        print("\nStarting battle visualization...")
        print("(Close the pygame window to stop)")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            while True:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Print current state
                print(f"\nStep {step_count + 1}")
                print(f"Agent HP: {obs[0]:.1f}, Bandit1 HP: {obs[1]:.1f}, Bandit2 HP: {obs[2]:.1f}")
                print(f"Agent Potions: {obs[3]:.1f}, Bandit1 Potions: {obs[4]:.1f}, Bandit2 Potions: {obs[5]:.1f}")
                print(f"Action taken: {['Attack Bandit1', 'Attack Bandit2', 'Use Potion'][action]}")
                
               
                # Take action
                new_obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                step_count += 1
                
                if done:
                    print(f"\nEpisode {episode + 1} finished!")
                    print(f"Final State - Agent HP: {new_obs[0]:.1f}, Bandit1 HP: {new_obs[1]:.1f}, Bandit2 HP: {new_obs[2]:.1f}")
                    print(f"Episode Reward: {episode_reward:.1f}")
                    print(f"Steps taken: {step_count}")
                    if env.agent_hp <= 0:
                        losses += 1
                    else:
                        wins += 1
                    print(f'Number of wins: {wins}')
                    print(f'Number of losses: {losses}')
                    break
                    
                obs = new_obs
                
    finally:
        env.close()
        
if __name__ == "__main__":
   train_agent(total_timesteps=1000000, agent_strength=10, bandit_strength=6)
   #test_agent(num_episodes=50, agent_strength=10, bandit_strength=6)



