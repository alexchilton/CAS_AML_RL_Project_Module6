# from stable_baselines3.common.policies import ActorCriticPolicy
# from sb3_contrib import RecurrentPPO  # ✅ Use RecurrentPPO (Supports LSTMs)


# import sys
# import matplotlib.pyplot as plt
# import numpy as np

# from stable_baselines3 import PPO
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.utils import get_linear_fn
# import pygame
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import gymnasium as gym

# import torch
# import torch.nn as nn
# import numpy as np
# from sb3_contrib import RecurrentPPO

# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# from graphic_env import BattleEnv
# from metrics_plotter import plot_training_metrics

# from stable_baselines3.common.evaluation import evaluate_policy
# from tqdm import tqdm

# # Define GRU Feature Extractor
# class GRUPredictor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#         super().__init__(observation_space, features_dim)
        
#         n_input = int(np.prod(observation_space.shape))
#         self.gru_hidden_size = 128  # Hidden size
#         self.gru_layers = 1  # Single-layer GRU
        
#         self.input_layer = nn.Sequential(
#             nn.Linear(n_input, 256),
#             nn.SiLU(),  
#             nn.LayerNorm(256)
#         )
        
#         self.gru = nn.GRU(256, self.gru_hidden_size, num_layers=self.gru_layers, batch_first=True)

#         self.fc = nn.Sequential(
#             nn.Linear(self.gru_hidden_size, features_dim),
#             nn.LeakyReLU(0.1),  
#             nn.LayerNorm(features_dim)
#         )

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         x = self.input_layer(observations)
#         x, _ = self.gru(x.unsqueeze(0))  # Batch-first
#         x = self.fc(x.squeeze(0))
#         return x

# # Training Function
# def train_recurrent_ppo(total_timesteps=500000, agent_strength=10, bandit_strength=6):
#     """
#     Train the Recurrent PPO agent with GRU and frame stacking.
#     """

#     # Create a vectorized environment
#     env = make_vec_env(lambda: BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength), n_envs=4)
    
#     # Apply frame stacking to keep past observations in memory
#     env = VecFrameStack(env, n_stack=4)

#     policy_kwargs = dict(
#         features_extractor_class=GRUPredictor,
#         features_extractor_kwargs=dict(features_dim=256),
#         net_arch=[128, 128]  # Two-layer MLP after GRU
#     )

#     def ent_coef_schedule(progress_remaining):
#         return max(0.01, 0.06 * progress_remaining)  # Dynamic entropy decay

#     model = RecurrentPPO(
#         "MlpLstmPolicy",  
#         env, 
#         verbose=1, 
#         device='cuda',
#         learning_rate=1.5e-4,  
#         n_steps=8192,  # Increase update steps
#         batch_size=1024,  
#         n_epochs=30,  
#         gamma=0.98,  
#         gae_lambda=0.995,  
#         clip_range=0.3,  
#         clip_range_vf=0.1,  
#         vf_coef=0.6,  
#         ent_coef=0.06,  
#         normalize_advantage=True, 
#         max_grad_norm=0.5,  
#         policy_kwargs=policy_kwargs
#     )

#     # Training with logging
#     model.learn(
#         total_timesteps=total_timesteps,
#         progress_bar=True,
#         log_interval=10
#     )

#     # Save the model
#     model.save("recurrent_rpg_model")
#     print("Model saved as 'recurrent_rpg_model'")



from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib import RecurrentPPO  # ✅ Use RecurrentPPO (Supports LSTMs)


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
from metrics_plotter import plot_training_metrics

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

        self.fc = nn.Sequential(
            nn.Linear(self.gru_hidden_size, features_dim),
            nn.LeakyReLU(0.1),  
            nn.LayerNorm(features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(observations)
        x, _ = self.gru(x.unsqueeze(0))  # Batch-first
        x = self.fc(x.squeeze(0))
        return x


def train_agent(total_timesteps = 1000, agent_strength = 10, bandit_strength = 6):
    """
    Train the PPO agent with a GRU-based feature extractor.
    
    Args:
        total_timesteps (int): Number of timesteps to train for
        strength (int): Initial strength parameter for the environment
    """    
    # initialize the enviroment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
        
    policy_kwargs = dict(
        features_extractor_class=GRUPredictor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128, 128]  
    )

    lr_schedule = get_linear_fn(start=1.5e-4, end=2e-5, end_fraction=0.8)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device='cpu',
        learning_rate=lr_schedule,  
        n_steps=4096,   
        batch_size=512, 
        n_epochs=25, 
        gamma=0.95,  
        gae_lambda=0.99,  
        clip_range=0.2, 
        clip_range_vf=0.1,  
        vf_coef=0.7,  # increased from 0.5
        ent_coef=0.06,  # increased from 0.03
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
    model = PPO.load("graphic_rpg_model_best")
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



    
    # Test with different opponent strength
    # results_strong = test_recurrent_ppo(num_episodes=50, bandit_strength=12, 
    #                                     save_results=True, model_path="recurrent_rpg_model")

# def train_agent(total_timesteps=1000, agent_strength=10, bandit_strength=6):
#     """
#     Train the PPO agent with a GRU-based feature extractor.
#     """    
#     env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
      
#     policy_kwargs = dict(
#         features_extractor_class=GRUPredictor,
#         features_extractor_kwargs=dict(features_dim=256),
#         net_arch=[128, 128]  
#     )

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

#     original_stdout = sys.stdout
#     try:
#         with open('training_log.txt', 'w') as f:
#             sys.stdout = f
#             model.learn(
#                 total_timesteps=total_timesteps,
#                 progress_bar=True, 
#                 log_interval=10)
#     finally:
#         sys.stdout = original_stdout

#     with open('training_log.txt', 'r') as f: 
#         log_content = f.read()
#     plot_training_metrics(log_content)
    
#     model.save("graphic_rpg_model")
#     print(f"model saved as 'graphic_rpg_model'")


        
if __name__ == "__main__":
    # train_agent(total_timesteps=100000, agent_strength=10, bandit_strength=6)
    test_agent(num_episodes=10, agent_strength=10, bandit_strength=6)

    # results = test_recurrent_ppo(num_episodes=50, render=False, verbose=True)
    # train_recurrent_ppo(total_timesteps=200000, agent_strength=10, bandit_strength=6)

    # with open('training_log.txt', 'r') as f: 
    #     log_content = f.read()
    # plot_training_metrics(log_content)


# #    Load the existing trained model
#     env = BattleEnv(agent_strength=10, bandit_strength=6)
#     model = PPO.load("graphic_rpg_model_backup", env=env)  # Ensure the environment is passed
#     model.save("graphic_rpg_model_backup") 
#     # model.policy.optimizer.param_groups[0]['lr'] = 3e-5

#     # Continue training for additional 1M timesteps
#     model.learn(total_timesteps=200000, progress_bar=True, log_interval=10)

#     # Save updated model
#     model.save("graphic_rpg_model")  # Overwrites the previous model with new training



    # env = BattleEnv(agent_strength=10, bandit_strength=6)
    # new_model = PPO(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=get_linear_fn(start=2e-4, end=1e-5, end_fraction=0.8),
    #     n_steps=8192,  # 🔥 Bigger steps for better learning
    #     batch_size=1024,
    #     gamma=0.97,
    #     gae_lambda=0.99,
    #     ent_coef=0.08,
    #     vf_coef=0.85,  # 🔥 Make critic stronger
    #     clip_range=lambda x: 0.15,  # 🔥 More stable updates
    #     policy_kwargs=dict(net_arch=[256, 256]),
    #     verbose=1
    # )

    # new_model = RecurrentPPO(
    #     "MlpLstmPolicy",  # ✅ Use the correct LSTM policy
    #     env,
    #     learning_rate=lambda x: 2e-4 * (1 - x) + 1e-5 * x,  # Linear decay
    #     n_steps=8192,
    #     batch_size=1024,
    #     gamma=0.97,
    #     gae_lambda=0.99,
    #     ent_coef=0.08,
    #     vf_coef=0.85,
    #     clip_range=lambda x: 0.15,
    #     policy_kwargs=dict(
    #         net_arch=[256, 256],  # ✅ Standard architecture
    #         activation_fn=torch.nn.ReLU,
    #         lstm_hidden_size=256,  # ✅ LSTM Memory Size
    #         enable_critic_lstm=True,  # ✅ Ensure LSTM memory for value function
    #         shared_lstm=False,  # ✅ Separate LSTM for actor and critic
    #     ),
    #     verbose=1,
    # )

    # new_model.learn(total_timesteps=1_000_000, progress_bar=True)
    # new_model.save("graphic_rpg_model")

