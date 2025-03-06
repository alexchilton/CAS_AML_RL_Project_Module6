import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import argparse
from graphic_env import BattleEnv
from metrics_plotter import plot_training_metrics

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class GRUNetwork(nn.Module):
    """
    GRU-based neural network with increased capacity for Q-learning
    """
    def __init__(self, input_size=15, hidden_size=256, gru_layers=2, output_size=3):
        super(GRUNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        self.output_size = output_size
        
        # Enhanced feature extraction
        self.preprocess = nn.Sequential(
            nn.Linear(input_size, 128),  # Increased from 64
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128)
        )
        
        # Deeper GRU with more units
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        
        # Dueling architecture with more capacity
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),  # Increased from 64
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),  # Increased from 64
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Preprocess each state in the sequence
        x_flat = x.reshape(-1, self.input_size)
        x_processed = self.preprocess(x_flat)
        x_processed = x_processed.reshape(batch_size, seq_length, 128)
        
        # Process through GRU
        if hidden is None:
            gru_out, hidden = self.gru(x_processed)
        else:
            gru_out, hidden = self.gru(x_processed, hidden)
            
        # Get last GRU output
        last_output = gru_out[:, -1]
        
        # Dueling architecture
        value = self.value_stream(last_output)
        advantages = self.advantage_stream(last_output)
        
        # Combine value and advantages: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values, hidden


class ReplayBuffer:
    """
    Experience replay buffer that stores sequences of states
    """
    def __init__(self, capacity=100000, sequence_length=6):  # Increased capacity and sequence length
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length
    
    def add(self, state_sequence, action, reward, next_state, done):
        self.buffer.append((state_sequence, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        
        # Convert to separate batches
        state_seqs, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            torch.FloatTensor(np.array(state_seqs)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )
    
    def __len__(self):
        return len(self.buffer)


class GRUDQNAgent:
    """
    Agent using Deep Q-Learning with GRU for sequence processing
    """
    def __init__(
        self,
        state_size=15,
        action_size=3,
        hidden_size=256,  # Increased from 128
        gru_layers=2,
        sequence_length=6,  # Increased from 4
        batch_size=128,  # Increased from 64
        learning_rate=0.0002,  # Adjusted for better learning
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.998,
        target_update_freq=200,
        memory_size=100000  # Increased from 50000
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        
        # Initialize networks
        self.q_network = GRUNetwork(state_size, hidden_size, gru_layers, action_size)
        self.target_network = GRUNetwork(state_size, hidden_size, gru_layers, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(memory_size, sequence_length)
        
        # Initialize state sequence buffer
        self.state_sequence = deque(maxlen=sequence_length)
        self.reset_sequence()
        
        # Initialize training statistics
        self.train_step_counter = 0
        self.episode_rewards = []
        self.episode_losses = []
        self.win_history = []
    
    def reset_sequence(self):
        """Reset the state sequence at the beginning of each episode"""
        self.state_sequence = deque(maxlen=self.sequence_length)
        for _ in range(self.sequence_length):
            self.state_sequence.append(np.zeros(self.state_size))
    
    def update_sequence(self, state):
        """Add a new state to the sequence"""
        self.state_sequence.append(state)
    
    def get_current_sequence(self):
        """Get the current state sequence as a numpy array"""
        return np.array(list(self.state_sequence))
    
    def act(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy"""
        # Add current state to sequence
        self.update_sequence(state)
        state_seq = self.get_current_sequence()
        
        # Check valid actions (from state indices 9-11)
        valid_actions = state[9:12].astype(bool)
        
        # Epsilon-greedy action selection
        if not eval_mode and random.random() < self.epsilon:
            # Random action from valid actions only
            if np.any(valid_actions):
                valid_indices = np.where(valid_actions)[0]
                return int(np.random.choice(valid_indices))
            else:
                return random.randrange(self.action_size)
        
        # Convert sequence to tensor for network
        state_seq_tensor = torch.FloatTensor(state_seq).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            q_values, _ = self.q_network(state_seq_tensor)
            q_values = q_values.squeeze().numpy()
        
        # Mask invalid actions
        if np.any(valid_actions):
            # Set invalid actions to large negative value
            masked_q_values = q_values.copy()
            masked_q_values[~valid_actions] = -1e9
            return int(np.argmax(masked_q_values))
        else:
            # Fallback if somehow no actions are valid
            return int(np.argmax(q_values))
    
    def learn(self):
        """Update Q-network based on experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples
        
        # Sample batch from replay buffer
        state_seqs, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Create next state sequences for target Q-values
        next_state_seqs = state_seqs.clone()
        for i in range(self.batch_size):
            # Shift sequence by one step
            next_state_seqs[i, :-1] = state_seqs[i, 1:]
            next_state_seqs[i, -1] = next_states[i]
        
        # Compute Q-values for current states
        current_q_values, _ = self.q_network(state_seqs)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Double DQN implementation
        with torch.no_grad():
            # Select actions using online network
            next_q_values, _ = self.q_network(next_state_seqs)
            best_actions = next_q_values.max(1)[1].unsqueeze(1)
            
            # Evaluate actions using target network
            next_target_q_values, _ = self.target_network(next_state_seqs)
            next_target_values = next_target_q_values.gather(1, best_actions)
            
            # Compute target Q-values
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_target_values
        
        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        # Get current sequence before adding next_state
        state_sequence = self.get_current_sequence()
        self.memory.add(state_sequence, action, reward, next_state, done)
    
    def save(self, filename):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load model"""
        try:
            checkpoint = torch.load(filename)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filename}")
            return True
        except:
            print(f"Error loading model from {filename}")
            return False


def train_agent(total_timesteps=500000, agent_strength=10, bandit_strength=6, progress_bar=True):
    """
    Train a GRU-DQN agent in the battle environment
    
    Args:
        total_timesteps (int): Maximum number of timesteps to train for
        agent_strength (int): Strength parameter for the agent
        bandit_strength (int): Strength parameter for the bandits
        progress_bar (bool): Whether to show progress updates
    """
    # Create environment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
    
    # Initialize the GRU-DQN agent
    agent = GRUDQNAgent(
        state_size=15,
        action_size=3,
        hidden_size=256,
        gru_layers=2,
        sequence_length=6,
        batch_size=128,
        learning_rate=0.0002,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.998,
        target_update_freq=200,
        memory_size=100000
    )
    
    # Training metrics
    episodes = 0
    total_steps = 0
    wins = 0
    losses = 0
    episode_rewards = []
    win_rates = []
    
    # Open log file
    log_file = open('high_capacity_gru_dqn_training_log.txt', 'w')
    log_file.write("Episode,Timestep,Reward,Loss,WinRate,Epsilon\n")
    
    try:
        print("Starting high capacity GRU-DQN agent training...")
        
        # Training loop
        while total_steps < total_timesteps:
            episodes += 1
            state, _ = env.reset()
            agent.reset_sequence()
            
            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            episode_step = 0
            done = False
            
            # Episode loop
            while not done and total_steps < total_timesteps:
                # Select action
                action = agent.act(state)
                
                # Take action in environment
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Learn from experience
                loss = agent.learn()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1
                
                # Update metrics
                episode_reward += reward
                episode_step += 1
                total_steps += 1
                
                # Update state
                state = next_state
            
            # Episode is complete - update statistics
            if env.agent_hp <= 0:  # Agent lost
                losses += 1
            else:  # Agent won
                wins += 1
                
            win_rate = (wins / episodes) * 100
            win_rates.append(win_rate)
            episode_rewards.append(episode_reward)
            
            # Calculate average loss if any learning steps occurred
            avg_loss = episode_loss / max(1, loss_count)
            agent.episode_losses.append(avg_loss)
            
            # Log to file
            log_entry = f"{episodes},{total_steps},{episode_reward:.2f},{avg_loss:.6f},{win_rate:.2f},{agent.epsilon:.4f}\n"
            log_file.write(log_entry)
            log_file.flush()
            
            # Print progress
            if progress_bar and (episodes % 10 == 0 or episodes == 1):
                print(f"Episode {episodes} | Steps: {total_steps}/{total_timesteps} | "
                      f"Reward: {episode_reward:.2f} | Win Rate: {win_rate:.2f}% | "
                      f"Epsilon: {agent.epsilon:.4f}")
            
            # Save model periodically
            if episodes % 500 == 0:
                agent.save(f"high_capacity_gru_dqn_model_{episodes}.pt")
                print(f"Checkpoint saved at episode {episodes}")
        
        # Save final model
        agent.save("high_capacity_gru_dqn_model_final.pt")
        
        print(f"\nTraining complete! Episodes: {episodes}, Final win rate: {win_rate:.2f}%")
        
        # Plot training metrics using your plotter
        with open('high_capacity_gru_dqn_training_log.txt', 'r') as f:
            log_content = f.read()
        plot_training_metrics(log_content)
        
    except KeyboardInterrupt:
        # Handle user interruption
        print("\nTraining interrupted. Saving current model...")
        agent.save("high_capacity_gru_dqn_model_interrupted.pt")
    
    finally:
        # Clean up
        log_file.close()
        env.close()
    
    return agent


def test_agent(num_episodes=100, agent_strength=10, bandit_strength=6, model_path="high_capacity_gru_dqn_model_final.pt"):
    """
    Test a trained agent
    
    Args:
        num_episodes (int): Number of episodes to test
        agent_strength (int): Strength parameter for the agent
        bandit_strength (int): Strength parameter for the bandits
        model_path (str): Path to the saved model
    """
    # Create environment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
    
    # Create agent
    agent = GRUDQNAgent(
        state_size=15,
        action_size=3,
        hidden_size=256,
        gru_layers=2,
        sequence_length=6,
        batch_size=128
    )
    
    # Load trained model
    if not agent.load(model_path):
        print("Failed to load model. Testing with untrained agent.")
    
    # Testing metrics
    wins = 0
    total_reward = 0
    episode_lengths = []
    
    print(f"\nTesting high capacity agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        agent.reset_sequence()
        episode_reward = 0
        step_count = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # Get action from agent
            action = agent.act(state, eval_mode=True)
            
            # Print current state and action
            print(f"\nStep {step_count + 1}")
            print(f"Agent HP: {state[0]:.1f}, Bandit1 HP: {state[1]:.1f}, Bandit2 HP: {state[2]:.1f}")
            print(f"Agent Potions: {state[3]:.1f}, Bandit1 Potions: {state[4]:.1f}, Bandit2 Potions: {state[5]:.1f}")
            print(f"Action taken: {['Attack Bandit1', 'Attack Bandit2', 'Use Potion'][action]}")
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Update metrics
            episode_reward += reward
            step_count += 1
            state = next_state
            
            # Check if episode is done
            if done:
                print(f"\nEpisode {episode + 1} finished!")
                print(f"Final State - Agent HP: {state[0]:.1f}, Bandit1 HP: {state[1]:.1f}, Bandit2 HP: {state[2]:.1f}")
                print(f"Episode Reward: {episode_reward:.1f}")
                print(f"Steps taken: {step_count}")
                
                if env.agent_hp > 0:  # Agent won
                    wins += 1
                    print("Result: VICTORY! 🎉")
                else:
                    print("Result: DEFEAT! 💀")
                
                total_reward += episode_reward
                episode_lengths.append(step_count)
                
                # Update win rate
                win_rate = (wins / (episode + 1)) * 100
                print(f"Current win rate: {win_rate:.2f}% ({wins}/{episode + 1})")
    
    # Final statistics
    win_rate = (wins / num_episodes) * 100
    avg_reward = total_reward / num_episodes
    avg_length = sum(episode_lengths) / len(episode_lengths)
    
    print("\n===== Testing Results =====")
    print(f"Episodes: {num_episodes}")
    print(f"Win Rate: {win_rate:.2f}% ({wins}/{num_episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f} steps")
    
    return win_rate, avg_reward


def compare_to_random(num_episodes=100, agent_strength=10, bandit_strength=6, model_path="high_capacity_gru_dqn_model_final.pt"):
    """
    Compare trained agent against random policy
    
    Args:
        num_episodes (int): Number of episodes for each agent
        agent_strength (int): Strength parameter for the agent
        bandit_strength (int): Strength parameter for the bandits
        model_path (str): Path to the saved model
    """
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)
    
    # Test trained agent
    print("\n===== Testing Trained Agent =====")
    trained_win_rate, trained_avg_reward = test_agent(
        num_episodes=num_episodes, 
        agent_strength=agent_strength, 
        bandit_strength=bandit_strength,
        model_path=model_path
    )
    
    # Test random agent
    print("\n===== Testing Random Agent =====")
    random_wins = 0
    random_total_reward = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Random action selection (only valid actions)
            valid_actions = state[9:12].astype(bool)
            if np.any(valid_actions):
                valid_indices = np.where(valid_actions)[0]
                action = int(np.random.choice(valid_indices))
            else:
                action = random.randrange(3)
            
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        if env.agent_hp > 0:  # Agent won
            random_wins += 1
            
        random_total_reward += episode_reward
        
        if (episode + 1) % 10 == 0:
            random_win_rate = (random_wins / (episode + 1)) * 100
            print(f"Episode {episode + 1}/{num_episodes} | Win Rate: {random_win_rate:.2f}%")
    
    random_win_rate = (random_wins / num_episodes) * 100
    random_avg_reward = random_total_reward / num_episodes
    
    # Print comparison
    print("\n===== Comparison =====")
    print(f"High Capacity GRU-DQN Agent: {trained_win_rate:.2f}% win rate, {trained_avg_reward:.2f} avg reward")
    print(f"Random Agent: {random_win_rate:.2f}% win rate, {random_avg_reward:.2f} avg reward")
    print(f"Improvement: {trained_win_rate - random_win_rate:.2f}% win rate, {trained_avg_reward - random_avg_reward:.2f} reward")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test GRU-DQN agent for RPG battles")
    
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'compare'], default='train',
                        help='Mode: train, test, or compare against random')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Number of timesteps for training')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes for testing/comparison')
    parser.add_argument('--agent_strength', type=int, default=10,
                        help='Strength parameter for the agent')
    parser.add_argument('--bandit_strength', type=int, default=6,
                        help='Strength parameter for the bandits')
    parser.add_argument('--model_path', type=str, default='high_capacity_gru_dqn_model_final.pt',
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
    elif args.mode == 'compare':
        compare_to_random(
            num_episodes=args.episodes,
            agent_strength=args.agent_strength,
            bandit_strength=args.bandit_strength,
            model_path=args.model_path
        )