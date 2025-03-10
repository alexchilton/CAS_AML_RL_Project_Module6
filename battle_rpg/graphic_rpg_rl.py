from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib import RecurrentPPO  # ✅ Use RecurrentPPO (Supports LSTMs)


import sys
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.policies import ActorCriticPolicy
import pygame
import torch.nn as nn
import torch.nn.functional as F
import torch
import gymnasium as gym
from gymnasium.core import Wrapper

from graphic_env import BattleEnv
from metrics_plotter import plot_training_metrics


# GRU with attention mechanism
class GRUPredictor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input = int(np.prod(observation_space.shape))
        self.gru_hidden_size = 128  # Hidden size
        self.gru_layers = 1  # Number of GRU layers
        
        self.input_layer = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.SiLU(),  
            nn.LayerNorm(256)
        )
        
        self.gru = nn.GRU(256, self.gru_hidden_size, num_layers=self.gru_layers, batch_first=True)
        
        # Fix the attention dimensions to match GRU output
        self.attention = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.gru_hidden_size, features_dim),
            nn.LeakyReLU(0.1),  
            nn.LayerNorm(features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(observations)
        
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Process batch of sequences with GRU
        # GRU expects input of shape [batch_size, seq_len, input_size]
        # For non-sequence data, we use seq_len=1
        x = x.unsqueeze(1) if len(x.shape) == 2 else x
        
        # Run GRU
        gru_out, _ = self.gru(x)
        
        # gru_out shape: [batch_size, seq_len, hidden_size]
        # For attention to work properly, we need to be careful with dimensions
        batch_size = gru_out.size(0)
        
        # Reshape for attention if needed
        if gru_out.size(1) == 1:  # If sequence length is 1
            # Just squeeze out the sequence dimension
            gru_out = gru_out.squeeze(1)
            # Simply use the final GRU output
            x = self.fc(gru_out)
        else:
            # Apply attention over sequence dimension
            # Reshape attention to match batch_size x seq_len x 1
            attn_weights = F.softmax(self.attention(gru_out).squeeze(-1), dim=1).unsqueeze(-1)
            # Apply attention weights
            context = torch.sum(gru_out * attn_weights, dim=1)
            # Final projection
            x = self.fc(context)
            
        return x
  

class DataAugmentationWrapper(Wrapper):
    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        
        if np.random.random() < 0.3:  # 30% chance of augmentation
            next_state = self._augment_state(next_state)
            
        return next_state, reward, done, truncated, info
    
    def reset(self, **kwargs):
        # Forward all arguments to the base environment
        return self.env.reset(**kwargs)
    
    def _augment_state(self, state):
        augmented = state.copy()
        for i in range(3):  # HP indices
            noise = np.random.uniform(-0.05, 0.05)
            if augmented[i] > 0:
                augmented[i] = max(0.1, augmented[i] * (1 + noise))
        return augmented

def train_agent(total_timesteps = 1000, agent_strength = 10, bandit_strength = 6):
    """
    Train the PPO agent with a GRU-based feature extractor.
    
    Args:
        total_timesteps (int): Number of timesteps to train for
        strength (int): Initial strength parameter for the environment
    """    
    # initialize the enviroment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
    env = DataAugmentationWrapper(env)
        
    policy_kwargs = dict(
        features_extractor_class=GRUPredictor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 128]
    )

    lr_schedule = get_linear_fn(start=1.5e-4, end=3e-5, end_fraction=0.8)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device='cpu',
        learning_rate=lr_schedule,  
        n_steps=2048,   
        batch_size=512, 
        n_epochs=25, 
        gamma=0.98,  
        gae_lambda=0.98,  
        clip_range=0.2, 
        clip_range_vf=0.1,  
        vf_coef=0.7,  # increased from 0.5
        ent_coef=0.03,  # increased from 0.03
        normalize_advantage=True, 
        max_grad_norm=0.3,  
        policy_kwargs=policy_kwargs
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
    # train_agent(total_timesteps=500000, agent_strength=10, bandit_strength=6)
    test_agent(num_episodes=100, agent_strength=10, bandit_strength=6)

    # results = test_recurrent_ppo(num_episodes=50, render=False, verbose=True)
    # train_recurrent_ppo(total_timesteps=200000, agent_strength=10, bandit_strength=6)

    # with open('training_log.txt', 'r') as f: 
    #     log_content = f.read()
    # plot_training_metrics(log_content)


# #    Load the existing trained model
#     env = BattleEnv(agent_strength=10, bandit_strength=6)
#     env = DataAugmentationWrapper(env)
#     model = PPO.load("graphic_rpg_model", env=env)  # Ensure the environment is passed
#     model.save("graphic_rpg_model_backup") 
#     # model.policy.optimizer.param_groups[0]['lr'] = 3e-5

#     # Continue training for additional 1M timesteps
#     model.learn(total_timesteps=500000, progress_bar=True, log_interval=10)

#     # Save updated model
#     model.save("graphic_rpg_model")  # Overwrites the previous model with new training



