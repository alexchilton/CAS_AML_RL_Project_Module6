import gymnasium as gym  # Fixed typo
import numpy as np

from graphic_env import BattleEnv
#from graphic_visualizer import GameVisualizer  
from metrics_plotter import plot_training_metrics

class RandomAgent:
    def __init__(self, env):
        """
        A simple agent that takes random actions.
        Args:
            env (gym.Env): The environment to interact with.
        """
        self.env = env
        self.action_space = env.action_space

    def act(self, observation):
        """
        Select a random action.
        Args:
            observation: The current state of the environment (not used).
        Returns:
            Random action from the action space.
        """
        return self.action_space.sample()

    def test(self, num_episodes=100):
        """
        Run the agent for a number of episodes and track win/loss statistics.
        Args:
            num_episodes (int): Number of episodes to run.
        """
        wins = 0
        losses = 0

        for episode in range(num_episodes):
            observation, _ = self.env.reset()  # Updated reset() for Gymnasium
            done = False

            while not done:
                action = self.act(observation)
                observation, reward, done, _, info = self.env.step(action)  # Updated step() for Gymnasium

            if reward > 0:  # Assuming reward > 0 means a win
                wins += 1
            else:
                losses += 1

        print(f"Random Agent Results - Wins: {wins}, Losses: {losses}")
        return wins, losses
    
if __name__ == "__main__":
    env = BattleEnv(agent_strength=10, bandit_strength=6)  # Initialize environment
    random_agent = RandomAgent(env)  # Create random agent

    print("Testing Random Agent...")
    random_wins, random_losses = random_agent.test(num_episodes=100)

    print(f"Random Agent Win Rate: {random_wins / (random_wins + random_losses):.2%}")

