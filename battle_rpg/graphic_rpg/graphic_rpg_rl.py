
import sys
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
import pygame

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
        clip_range=0.1, 
        normalize_advantage=True, 
        vf_coef = 0.3
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
                    break
                    
                obs = new_obs
                
    finally:
        env.close()
        
if __name__ == "__main__":
#   train_agent(total_timesteps=100000, agent_strength=10, bandit_strength=6)
    test_agent(num_episodes=2, agent_strength=10, bandit_strength=6)



