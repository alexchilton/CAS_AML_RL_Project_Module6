
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
    Train the PPO agent with an LSTM-based feature extractor.
    
    Args:
        total_timesteps (int): Number of timesteps to train for
        strength (int): Initial strength parameter for the environment
    """    
    # initialize the enviroment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
    
    # class LSTMPredictor(BaseFeaturesExtractor):
    #     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
    #         super().__init__(observation_space, features_dim)
            
    #         n_input = int(np.prod(observation_space.shape))
    #         self.lstm_hidden_size = 256  # Increased memory size
    #         self.lstm_layers = 2  # Added an extra LSTM layer
            
    #         self.input_layer = nn.Sequential(
    #             nn.Linear(n_input, 256),
    #             nn.SiLU(),  # Swish activation for smoother gradient flow
    #             nn.LayerNorm(256)
    #         )
            
    #         self.lstm = nn.LSTM(256, self.lstm_hidden_size, num_layers=self.lstm_layers, batch_first=True)
            
    #         self.fc = nn.Sequential(
    #             nn.Linear(self.lstm_hidden_size, features_dim),
    #             nn.SiLU(),
    #             nn.LayerNorm(features_dim)
    #         )

    #     def forward(self, observations: torch.Tensor) -> torch.Tensor:
    #         x = self.input_layer(observations)
    #         x, _ = self.lstm(x.unsqueeze(0))  # Keep batch_first=True
    #         x = self.fc(x.squeeze(0))
    #         return x
    
    # policy_kwargs = dict(
    #     features_extractor_class=LSTMPredictor,
    #     features_extractor_kwargs=dict(features_dim=256),
    #     net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    # )
       
    class LSTMPredictor(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            
            n_input = int(np.prod(observation_space.shape))
            self.lstm_hidden_size = 128  # Reduced from 256 to 128
            self.lstm_layers = 1  # Only 1 LSTM layer for simplicity
            
            self.input_layer = nn.Sequential(
                nn.Linear(n_input, 256),
                nn.SiLU(),  
                nn.LayerNorm(256)
            )
            
            self.lstm = nn.LSTM(256, self.lstm_hidden_size, num_layers=self.lstm_layers, batch_first=True)

            self.fc = nn.Sequential(
                nn.Linear(self.lstm_hidden_size, features_dim),
                nn.LeakyReLU(0.1),  
                nn.LayerNorm(features_dim)
            )

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            x = self.input_layer(observations)
            x, _ = self.lstm(x.unsqueeze(0))  # Keep batch_first=True
            x = self.fc(x.squeeze(0))
            return x

    policy_kwargs = dict(
        features_extractor_class=LSTMPredictor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128, 128]  # Reduced from 256, 256
    )


    # initialize the PPO model 
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device='cpu',
        learning_rate=1e-4,  # reduced in switch Resnet --> LSTM
        n_steps=2048,   # reduced in switch Resnet --> LSTM
        batch_size=512, 
        n_epochs=25, # increased in switch Resnet --> LSTM
        gamma=0.95,  # adjusted for temporal credit assignment
        gae_lambda=0.98,  # adjusted for temporal credit assignment
        clip_range=0.2, 
        clip_range_vf=0.1,  # Add value function clipping
        vf_coef = 0.5,
        ent_coef= 0.02,   # explore smarter exploration
        normalize_advantage=True, 
        max_grad_norm=0.3,  # prevent explosing gradient
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
    Test the trained agent on LSTM-based feature extractor.
    
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
   #test_agent(num_episodes=5, agent_strength=10, bandit_strength=6)



