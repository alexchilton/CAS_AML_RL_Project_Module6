
import sys
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn
import pygame
import torch.nn as nn
import torch.nn.functional as F
import torch
import gymnasium as gym

from graphic_env import BattleEnv
#from graphic_visualizer import GameVisualizer  
from metrics_plotter import plot_training_metrics

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


def train_agent(total_timesteps = 1000, agent_strength = 10, bandit_strength = 6):
    """
    Train the PPO agent with an LSTM-based feature extractor.
    
    Args:
        total_timesteps (int): Number of timesteps to train for
        strength (int): Initial strength parameter for the environment
    """    
    # initialize the enviroment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
      
    policy_kwargs = dict(
        features_extractor_class=LSTMPredictor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128, 128]  # Reduced from 256, 256
    )


    # initialize the PPO model 
    lr_schedule = get_linear_fn(start=1e-4, end=1e-5, end_fraction=0.8)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device='cpu',
        learning_rate=lr_schedule,  
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



# def train_agent(total_timesteps=1000, agent_strength=10, bandit_strength=6, progress_bar=True):
#     """
#     Train the PPO agent with an LSTM-based feature extractor.
    
#     Args:
#         total_timesteps (int): Number of timesteps to train for.
#         agent_strength (int): Initial strength parameter for the environment.
#         bandit_strength (int): Initial strength parameter for the enemy.
#         progress_bar (bool): Whether to show the progress bar.
#     """    
#     # Initialize the environment
#     env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)

#     policy_kwargs = dict(
#         features_extractor_class=LSTMPredictor,
#         features_extractor_kwargs=dict(features_dim=256),
#         net_arch=[128, 128]
#     )

#     # Initialize the PPO model
#     lr_schedule = get_linear_fn(start=1e-4, end=1e-5, end_fraction=0.8)
#     model = PPO(
#         "MlpPolicy", 
#         env, 
#         verbose=1, 
#         device='cpu',
#         learning_rate=lr_schedule,  
#         n_steps=2048,
#         batch_size=512, 
#         n_epochs=25,
#         gamma=0.95,  
#         gae_lambda=0.98,  
#         clip_range=0.2, 
#         clip_range_vf=0.1,  
#         vf_coef=0.5,
#         ent_coef=0.02,   
#         normalize_advantage=True, 
#         max_grad_norm=0.3,  
#         policy_kwargs=policy_kwargs
#     )

#     # Capture logs but keep the progress bar visible
#     with open('training_log.txt', 'w') as f:
#         original_stdout = sys.stdout
#         sys.stdout = original_stdout  # Reset stdout to allow progress bar

#         try:
#             model.learn(
#                 total_timesteps=total_timesteps,
#                 progress_bar=progress_bar,  # ✅ Progress bar will be shown
#                 log_interval=10
#             )
#         finally:
#             sys.stdout = original_stdout  # Restore original stdout

#     # Plot metrics
#     with open('training_log.txt', 'r') as f:
#         log_content = f.read()
#     plot_training_metrics(log_content)
    
#     # Save the model
#     model.save("graphic_rpg_model")
#     print(f"Model saved as 'graphic_rpg_model'")







def test_agent(num_episodes=5, agent_strength=10, bandit_strength=6):
    """
    Test the trained agent on LSTM-based feature extractor.
    
    Args:
        num_episodes (int): Number of episodes to test
        agent_strength (int): Strength parameter for the agent
        bandit_strength (int): Strength parameter for the bandits
    """
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
    model = PPO.load("graphic_rpg_model_retrained")
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

# #    Load the existing trained model
#     env = BattleEnv(agent_strength=10, bandit_strength=6)
#     model = PPO.load("graphic_rpg_model", env=env)  # Ensure the environment is passed
#     model.save("graphic_rpg_model_backup") 
#     # model.policy.optimizer.param_groups[0]['lr'] = 3e-5

#     # Continue training for additional 1M timesteps
#     model.learn(total_timesteps=500000, progress_bar=True, log_interval=10)

#     # Save updated model
#     model.save("graphic_rpg_model")  # Overwrites the previous model with new training






    # # Retraining from a base trained model 

    # env = BattleEnv(agent_strength=10, bandit_strength=6)

    # # Load model but reset optimizer
    # old_model = PPO.load("graphic_rpg_model", env=env)
    # new_model = PPO("MlpPolicy", env, policy_kwargs=old_model.policy_kwargs, verbose=1)

    # # Copy weights from the old model to the new model
    # new_model.policy.load_state_dict(old_model.policy.state_dict())

    # # Train again from this state
    # new_model.learn(total_timesteps=500000, progress_bar=True, log_interval=10)
    # new_model.save("graphic_rpg_model_retrained")






    # env = BattleEnv(agent_strength=10, bandit_strength=6)

    # # Load the previously trained model
    # old_model = PPO.load("graphic_rpg_model", env=env)

    # # Use the custom policy architecture and training parameters
    # policy_kwargs = dict(
    #     features_extractor_class=LSTMPredictor,
    #     features_extractor_kwargs=dict(features_dim=256),
    #     net_arch=[128, 128]
    # )

    # lr_schedule = get_linear_fn(start=1e-4, end=1e-5, end_fraction=0.8)

    # # Create a new model with the same architecture and load old weights
    # new_model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     device='cpu',
    #     learning_rate=lr_schedule,
    #     n_steps=2048,
    #     batch_size=512,
    #     n_epochs=10,
    #     gamma=0.98,
    #     gae_lambda=0.98,
    #     clip_range=0.2,
    #     clip_range_vf=0.1,
    #     vf_coef=0.5,
    #     ent_coef=0.02,
    #     normalize_advantage=True,
    #     max_grad_norm=0.3,
    #     policy_kwargs=policy_kwargs
    # )

    # # Load weights from the old model
    # new_model.policy.load_state_dict(old_model.policy.state_dict())

    # # Continue training
    # new_model.learn(total_timesteps=500000, progress_bar=True, log_interval=10)

    # # Save the updated model
    # new_model.save("graphic_rpg_model_retrained")


