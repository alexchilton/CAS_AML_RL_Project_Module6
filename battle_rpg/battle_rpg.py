
import sys
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
import pygame

from battle_env import BattleEnv
from battle_visualizer import BattleVisualizer
from metrics_plotter import plot_training_metrics

def train_agent():
    env = BattleEnv()
    model = PPO("MlpPolicy", env, verbose=1, device='cpu')

    # to output file
    original_stdout=sys.stdout
    with open('training_log.txt', 'w') as f:
        sys.stdout=f
        model.learn(total_timesteps=100000)
    sys.stdout = original_stdout

    # plot metrics
    with open('training_log.txt', 'r') as f: 
        log_content=f.read()
    plot_training_metrics(log_content)
    
    # Save the model
    model.save("battle_model")

# def test_agent():
#     env = BattleEnv()
#     model = PPO.load("battle_model")
#     visualizer=BattleVisualizer()
    
#     obs, _ = env.reset()
#     done = False
#     total_reward = 0
    
#     try:
#         print("\nStarting battle visualization...")
#         print("(Close the pygame window to stop)")
        
#         while True: 
#             action, _ = model.predict(obs)
#             print(f"\nAgent HP: {obs[0]:.1f}, Boss HP: {obs[1]:.1f}")
#             print(f"Action: {action}")
            
#             visualizer.visualize_state(obs, action)
#             new_obs, reward, done, truncated, _ = env.step(action)
#             total_reward += reward
            
#             if done:  
#                 print(f"\nFinal State - Agent HP: {new_obs[0]:.1f}, Boss HP: {new_obs[1]:.1f}")
#                 print(f"Last Action: {action}")
#                 visualizer.visualize_state(new_obs, action)  
#                 print(f"\nBattle finished! Total reward: {total_reward:.1f}")
#                 break
                
#             obs = new_obs
        
#         # Keep window open until we closes it
#         running = True
#         while running:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#                     break
#             pygame.display.flip()
            
#     except KeyboardInterrupt:
#         print("\nVisualization stopped by user")
#     finally:
#         visualizer.close()
        
# if __name__ == "__main__":
#     train_agent()
#     test_agent()


def test_agent(human_controlled=False):
    env = BattleEnv(human_controlled=human_controlled)  # Pass human_controlled parameter
    model = PPO.load("battle_model")
    visualizer = BattleVisualizer(human_controlled=human_controlled)
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    print("\nStarting battle...")
    if human_controlled:
        print("(You are the Boss! Enter numbers 0-3 to select actions)")
    
    try:
        while not done:
            # Get agent's action
            action_agent, _ = model.predict(obs)
            
            # Step the environment with agent's action
            new_obs, reward, done, truncated, _ = env.step(action_agent)
            total_reward += reward
            
            # Visualize the battle state
            visualizer.visualize_state(new_obs, action_agent)
            
            # Print current state
            print(f"\nAgent HP: {new_obs[0]:.1f}, Boss HP: {new_obs[1]:.1f}")
            print(f"Agent Action: {action_agent}")
            
            if done:
                print(f"\nFinal State - Agent HP: {new_obs[0]:.1f}, Boss HP: {new_obs[1]:.1f}")
                print(f"Battle finished! Total reward: {total_reward:.1f}")
                break
            
            obs = new_obs
            
        # Keep window open until closed
        if not human_controlled:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                pygame.display.flip()
                
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        visualizer.close()

if __name__ == "__main__":
    #train_agent()
    test_agent(human_controlled=False)
