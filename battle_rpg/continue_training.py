# Create this script to continue training
import pickle
import numpy as np
from graphic_rpg_rl_qlearning import train_agent, QlearningAgent, discretize_state
from graphic_env import BattleEnv

# Load existing model
with open("graphic_rpg_qlearning_model.pkl", 'rb') as f:
    q_table = pickle.load(f)

# Create environment
env = BattleEnv(agent_strength=10, bandit_strength=6)

# Initialize agent with the loaded Q-table
agent = QlearningAgent(
    state_space_size=324,
    action_space_size=3,
    learning_rate=0.2,
    discount_factor=0.99,
    exploration_rate=0.3,  # Reset exploration rate to encourage finding new strategies
    min_exploration_rate=0.01,
    exploration_decay=0.9995  # Slower decay
)
agent.q_table = q_table  # Load the previously trained Q-table

# Continue training
episodes = 0
total_steps = 0
wins = 0
losses = 0
additional_steps = 50000  # Train for another 50,000 steps

# Open a log file to continue logging
with open('qlearning_training_log_continued.txt', 'w') as log_file:
    log_file.write("Episode,Reward,Loss,WinRate,Epsilon\n")

    # Training loop
    while total_steps < additional_steps:
        episodes += 1

        # Reset the environment
        obs, _ = env.reset()
        state = discretize_state(obs)
        episode_reward = 0
        episode_loss = 0
        steps = 0
        done = False

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

            next_state = discretize_state(next_obs)
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

        # End of episode
        agent.update_exploration_rate()

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
            print(f"Episode {episodes} | Steps: {total_steps}/{additional_steps} | "
                  f"Reward: {episode_reward:.2f} | Win Rate: {win_rate:.2f}% | "
                  f"Epsilon: {agent.exploration_rate:.4f}")

        # Periodically bump up exploration to escape local optima
        if episodes % 500 == 0:
            agent.exploration_rate = max(0.2, agent.exploration_rate)
            print(f"Exploration reset to {agent.exploration_rate}")

        # Save model at intervals
        if episodes % 500 == 0:
            agent.save(f"graphic_rpg_qlearning_model_continued_{episodes}.pkl")

    # Save final model
    agent.save("graphic_rpg_qlearning_model_improved.pkl")
    print("Training complete! Improved model saved.")