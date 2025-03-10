import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

from graphic_env import BattleEnv
from metrics_plotter import plot_training_metrics

# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class QlearningAgent:
    """
    Q-learning agent for the battle environment.
    This uses a tabular approach with a state-action table.
    """

    def __init__(self, state_space_size, action_space_size,
                 learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, min_exploration_rate=0.01,
                 exploration_decay=0.995):

        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_space_size, action_space_size))

        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay

        # For tracking learning progress
        self.rewards_history = []
        self.epsilon_history = []
        self.loss_history = []

    def get_action(self, state, valid_actions=None):
        """
        Select an action using epsilon-greedy policy

        Args:
            state: Current state index
            valid_actions: List of valid actions (optional)

        Returns:
            action: Selected action
        """
        # Exploration
        if np.random.random() < self.exploration_rate:
            if valid_actions is not None and sum(valid_actions) > 0:
                # Choose from valid actions for exploration
                valid_indices = np.where(valid_actions)[0]
                return np.random.choice(valid_indices)
            else:
                # Fallback to random from all actions
                return np.random.randint(self.q_table.shape[1])

        # Exploitation
        q_values = self.q_table[state, :]

        # If we have valid actions info, mask invalid actions
        if valid_actions is not None:
            # Create a masked version of q_values
            masked_q_values = q_values.copy()

            # Set invalid actions to very negative values
            if sum(valid_actions) > 0:  # If there are valid actions
                for i, is_valid in enumerate(valid_actions):
                    if not is_valid:
                        masked_q_values[i] = -np.inf

                return np.argmax(masked_q_values)

        # Fallback to normal argmax if no valid actions info
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done, valid_actions_next=None):
        """
        Update Q-values using the Q-learning update rule

        Args:
            state: Current state index
            action: Action taken
            reward: Reward received
            next_state: Next state index
            done: Whether episode is done
            valid_actions_next: Valid actions for next state
        """
        # Get best next action based on valid actions
        if valid_actions_next is not None and sum(valid_actions_next) > 0:
            # Mask invalid actions for next state
            next_q_values = self.q_table[next_state, :].copy()
            for i, is_valid in enumerate(valid_actions_next):
                if not is_valid:
                    next_q_values[i] = -np.inf
            best_next_action = np.argmax(next_q_values)
        else:
            # If no valid actions info, use normal argmax
            best_next_action = np.argmax(self.q_table[next_state, :])

        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * self.q_table[next_state, best_next_action]

        # Calculate loss (for monitoring only)
        current_q = self.q_table[state, action]
        loss = (target_q - current_q) ** 2

        # Update Q-value
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)

        return loss

    def update_exploration_rate(self):
        """Decay exploration rate"""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        self.epsilon_history.append(self.exploration_rate)

    def save(self, filepath):
        """Save the Q-table to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved as '{filepath}'")

    def load(self, filepath):
        """Load the Q-table from a file"""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Model loaded from '{filepath}'")


def discretize_state(obs):
    """
    Convert the continuous observation to a discrete state index
    for the Q-table. This is a simple binning approach.

    Args:
        obs: Observation from environment (numpy array)

    Returns:
        state_index: Discrete state index
    """
    # Extract key information from the observation
    agent_hp = obs[0]
    bandit1_hp = obs[1]
    bandit2_hp = obs[2]
    agent_potions = obs[3]

    # Discretize HP values into bins (low, medium, high)
    agent_hp_bin = 0 if agent_hp < 10 else (1 if agent_hp < 20 else 2)
    bandit1_hp_bin = 0 if bandit1_hp <= 0 else (1 if bandit1_hp < 10 else 2)
    bandit2_hp_bin = 0 if bandit2_hp <= 0 else (1 if bandit2_hp < 10 else 2)

    # Discretize potion count
    agent_potions_bin = min(int(agent_potions), 3)

    # Add binary information about which bandit has lower HP (if both are alive)
    weaker_bandit = 0  # 0 = both equal or both dead
    if bandit1_hp > 0 and bandit2_hp > 0:
        weaker_bandit = 1 if bandit1_hp < bandit2_hp else 2  # 1 = bandit1 weaker, 2 = bandit2 weaker
    elif bandit1_hp > 0:
        weaker_bandit = 2  # only bandit1 alive, so focus on bandit2 (which is dead)
    elif bandit2_hp > 0:
        weaker_bandit = 1  # only bandit2 alive, so focus on bandit1 (which is dead)

    # Combine bins into a single state index
    # 3 states for agent HP × 3 states for bandit1 HP × 3 states for bandit2 HP × 4 states for potions × 3 states for weaker bandit = 324 states
    state_index = (
            agent_hp_bin * 108 +
            bandit1_hp_bin * 36 +
            bandit2_hp_bin * 12 +
            agent_potions_bin * 3 +
            weaker_bandit
    )

    return state_index


def train_agent(total_timesteps=1000000, agent_strength=10, bandit_strength=6,
                learning_rate=0.2, save_interval=1000, render=False):
    """
    Train a Q-learning agent in the battle environment

    Args:
        total_timesteps: Maximum number of timesteps to train for
        agent_strength: Strength parameter for the agent
        bandit_strength: Strength parameter for the bandits
        learning_rate: Learning rate for Q-learning updates
        save_interval: How often to save the model (in episodes)
        render: Whether to render the environment during training
    """
    # Create the environment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)

    # State space size (from our discretization function)
    state_space_size = 324  # 3×3×3×4×3 = 324 discrete states

    # Action space size
    action_space_size = 3  # Attack bandit 1, Attack bandit 2, Use potion

    # Initialize Q-learning agent
    agent = QlearningAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=learning_rate,  # Increased from 0.1 to 0.2
        discount_factor=0.99,  # Increased from 0.95 to 0.99 to value future rewards more
        exploration_rate=1.0,
        min_exploration_rate=0.01,
        exploration_decay=0.998  # Slowed down from 0.995 to 0.998 for more thorough exploration
    )

    # Training metrics
    episodes = 0
    total_steps = 0
    wins = 0
    losses = 0
    all_rewards = []
    episode_losses = []

    # Open a log file for training metrics
    with open('qlearning_training_log.txt', 'w') as log_file:
        log_file.write("Episode,Reward,Loss,WinRate,Epsilon\n")

        # Training loop
        while total_steps < total_timesteps:
            episodes += 1

            # Reset the environment
            obs, _ = env.reset()
            state = discretize_state(obs)
            episode_reward = 0
            episode_loss = 0
            steps = 0
            done = False

            # Track damages dealt to each bandit for reward shaping
            bandit1_damage_dealt = 0
            bandit2_damage_dealt = 0

            # Previous health values to calculate damage
            prev_bandit1_hp = obs[1]
            prev_bandit2_hp = obs[2]

            # Episode loop
            while not done:
                # Get valid actions from the observation
                valid_actions = obs[9:12].astype(bool)

                # Get action from policy
                action = agent.get_action(state, valid_actions)

                # Take action in environment
                next_obs, reward, done, truncated, _ = env.step(action)

                # Calculate damage dealt to bandits for reward shaping
                damage_to_bandit1 = max(0, prev_bandit1_hp - next_obs[1])
                damage_to_bandit2 = max(0, prev_bandit2_hp - next_obs[2])

                # Track damage
                bandit1_damage_dealt += damage_to_bandit1
                bandit2_damage_dealt += damage_to_bandit2

                # Update previous health values
                prev_bandit1_hp = next_obs[1]
                prev_bandit2_hp = next_obs[2]

                # Implement reward shaping
                shaped_reward = reward

                # Strategic targeting reward: bonus for attacking the weaker bandit
                if action == 0 and next_obs[1] > 0 and next_obs[2] > 0:  # Attack bandit 1
                    if next_obs[1] < next_obs[2]:  # Bandit 1 has less HP
                        shaped_reward += 2.0  # Bonus for targeting weaker bandit

                elif action == 1 and next_obs[1] > 0 and next_obs[2] > 0:  # Attack bandit 2
                    if next_obs[2] < next_obs[1]:  # Bandit 2 has less HP
                        shaped_reward += 2.0  # Bonus for targeting weaker bandit

                # Reward for killing a bandit
                if prev_bandit1_hp > 0 and next_obs[1] <= 0:
                    shaped_reward += 10.0  # Big bonus for killing a bandit

                if prev_bandit2_hp > 0 and next_obs[2] <= 0:
                    shaped_reward += 10.0  # Big bonus for killing a bandit

                # Potion usage reward - encourage using potions when HP is low
                if action == 2:  # Use potion
                    if obs[0] < 10:  # HP is low
                        shaped_reward += 3.0  # Good timing for using potion
                    elif obs[0] > 20:  # HP is high
                        shaped_reward -= 1.0  # Penalty for using potion when not needed

                # Process next state
                next_state = discretize_state(next_obs)

                # Valid actions for next state
                valid_actions_next = next_obs[9:12].astype(bool)

                # Learn from experience with shaped reward
                loss = agent.learn(state, action, shaped_reward, next_state, done, valid_actions_next)

                # Update metrics
                episode_reward += reward  # Track original reward for logging
                episode_loss += loss
                state = next_state
                obs = next_obs
                steps += 1
                total_steps += 1

            # End of episode processing
            agent.update_exploration_rate()

            # Record metrics
            agent.rewards_history.append(episode_reward)
            all_rewards.append(episode_reward)
            episode_losses.append(episode_loss / steps)
            agent.loss_history.append(episode_loss / steps)

            # Track wins/losses
            if env.agent_hp <= 0:  # Agent lost
                losses += 1
            else:  # Agent won
                wins += 1

            win_rate = (wins / episodes) * 100

            # Log to file
            log_file.write(
                f"{episodes},{episode_reward:.2f},{episode_loss / steps:.6f},{win_rate:.2f},{agent.exploration_rate:.4f}\n")

            # Print progress
            if episodes % 10 == 0:
                print(f"Episode {episodes} | Steps: {total_steps}/{total_timesteps} | "
                      f"Reward: {episode_reward:.2f} | Win Rate: {win_rate:.2f}% | "
                      f"Epsilon: {agent.exploration_rate:.4f}")

            # Save model at intervals
            if episodes % save_interval == 0:
                agent.save(f"graphic_rpg_qlearning_model_{episodes}.pkl")

    # Save final model
    agent.save("graphic_rpg_qlearning_model.pkl")

    # Plot training metrics
    with open('qlearning_training_log.txt', 'r') as f:
        log_content = f.read()
    plot_training_metrics(log_content)

    # Additional custom plots
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(agent.rewards_history)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(agent.epsilon_history)
    plt.title('Exploration Rate Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('qlearning_training_plots.png')
    plt.show()

    return agent


def test_agent(num_episodes=5, agent_strength=10, bandit_strength=6, model_path="graphic_rpg_qlearning_model.pkl"):
    """
    Test a trained Q-learning agent

    Args:
        num_episodes: Number of episodes to test
        agent_strength: Strength parameter for the agent
        bandit_strength: Strength parameter for the bandits
        model_path: Path to the saved model
    """
    # Create the environment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)

    # State space and action space sizes
    state_space_size = 324  # Same as in training
    action_space_size = 3  # Same as in training

    # Initialize Q-learning agent
    agent = QlearningAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size
    )

    # Load trained model
    agent.load(model_path)

    # Set exploration rate to minimum for testing (mostly exploitation)
    agent.exploration_rate = 0.01

    # Track wins and total rewards
    wins = 0
    total_rewards = 0

    print(f"\nTesting agent over {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Reset environment
        obs, _ = env.reset()
        state = discretize_state(obs)
        episode_reward = 0
        step_count = 0
        done = False

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not done:
            # Get valid actions
            valid_actions = obs[9:12].astype(bool)

            # Select action
            action = agent.get_action(state, valid_actions)

            # Print current state and action
            print(f"\nStep {step_count + 1}")
            print(f"Agent HP: {obs[0]:.1f}, Bandit1 HP: {obs[1]:.1f}, Bandit2 HP: {obs[2]:.1f}")
            print(f"Agent Potions: {obs[3]:.1f}, Bandit1 Potions: {obs[4]:.1f}, Bandit2 Potions: {obs[5]:.1f}")
            print(f"Action taken: {['Attack Bandit1', 'Attack Bandit2', 'Use Potion'][action]}")

            # Take action
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs)

            # Update for next step
            obs = next_obs
            state = next_state
            episode_reward += reward
            step_count += 1

            # Check if episode is done
            if done:
                print(f"\nEpisode {episode + 1} finished!")
                print(f"Final State - Agent HP: {obs[0]:.1f}, Bandit1 HP: {obs[1]:.1f}, Bandit2 HP: {obs[2]:.1f}")
                print(f"Episode Reward: {episode_reward:.1f}")
                print(f"Steps taken: {step_count}")

                if env.agent_hp > 0:  # Agent won
                    wins += 1
                    print("Result: VICTORY! 🎉")
                else:
                    print("Result: DEFEAT! 💀")

                total_rewards += episode_reward

    # Print final results
    win_rate = (wins / num_episodes) * 100
    avg_reward = total_rewards / num_episodes

    print("\n===== Test Results =====")
    print(f"Episodes: {num_episodes}")
    print(f"Win Rate: {win_rate:.2f}% ({wins}/{num_episodes})")
    print(f"Average Reward: {avg_reward:.2f}")

    return win_rate, avg_reward


def test_visualizer(num_episodes=2, agent_strength=10, bandit_strength=6, model_path="graphic_rpg_qlearning_model.pkl"):
    """
    Run the battle visualization using the Q-learning model.
    This function is designed to be compatible with the existing visualization code.

    Args:
        num_episodes: Number of episodes to visualize
        agent_strength: Strength parameter for the agent
        bandit_strength: Strength parameter for the bandits
        model_path: Path to the trained model
    """
    try:
        # Import here to avoid circular imports
        from graphic_visualizer import GameVisualizer

        # Initialize agent
        state_space_size = 324
        action_space_size = 3
        agent = QlearningAgent(state_space_size, action_space_size)

        # Try to load the model, or create a new one if file doesn't exist
        try:
            agent.load(model_path)
        except FileNotFoundError:
            print(f"Model file '{model_path}' not found. Testing with untrained agent.")

        # Set exploration to minimum for visualization
        agent.exploration_rate = 0.01

        # Initialize the visualizer
        visualizer = GameVisualizer(agent_strength=agent_strength, bandit_strength=bandit_strength)

        # Run visualization
        try:
            print("\nStarting battle visualization with Q-learning agent...")
            print("(Close the pygame window to stop)")

            # This is the main visualization loop
            # It's designed to be compatible with the existing visualization code
            visualizer.run_visualization_qlearning(agent, discretize_state, num_episodes=num_episodes)

        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()

    except ImportError:
        print("Could not import visualization module. Make sure graphic_visualizer.py is in the same directory.")


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test Q-learning agent for RPG battles")

    parser.add_argument('--mode', type=str, choices=['train', 'test', 'visualize'], default='train',
                        help='Mode: train, test, or visualize')
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Number of timesteps for training')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes for testing/visualization')
    parser.add_argument('--agent_strength', type=int, default=10,
                        help='Strength parameter for the agent')
    parser.add_argument('--bandit_strength', type=int, default=6,
                        help='Strength parameter for the bandits')
    parser.add_argument('--model_path', type=str, default='graphic_rpg_qlearning_model.pkl',
                        help='Path to load/save model')

    args = parser.parse_args()

    if args.mode == 'train':
        train_agent(
            total_timesteps=args.timesteps,
            agent_strength=args.agent_strength,
            bandit_strength=args.bandit_strength
        )
    elif args.mode == 'test':
        test_agent(
            num_episodes=args.episodes,
            agent_strength=args.agent_strength,
            bandit_strength=args.bandit_strength,
            model_path=args.model_path
        )
    elif args.mode == 'visualize':
        test_visualizer(
            num_episodes=args.episodes,
            agent_strength=args.agent_strength,
            bandit_strength=args.bandit_strength,
            model_path=args.model_path
        )