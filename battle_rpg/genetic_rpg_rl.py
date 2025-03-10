import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from tqdm import tqdm
import os
import copy

from graphic_env import BattleEnv
from metrics_plotter import plot_training_metrics

# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class GeneticPolicyIndividual:
    """
    Represents a single individual (policy) in the genetic algorithm population.
    Each individual is a decision tree for determining actions in the battle environment.
    """

    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.fitness = 0  # Will store the individual's fitness
        self.decision_tree = self.generate_random_tree(depth=0)
        self.wins = 0
        self.total_games = 0

    def get_action(self, state_dict, valid_actions=None):
        """
        Get action from the decision tree based on the current state

        Args:
            state_dict: Dictionary containing state information
            valid_actions: List of valid actions

        Returns:
            action: Selected action
        """
        # If we have valid actions information, mask invalid actions
        if valid_actions is not None and sum(valid_actions) == 0:
            # No valid actions, return random
            return random.randint(0, 2)

        # Safely navigate through decision tree
        current_node = self.decision_tree

        while True:
            # Ensure node is a valid dictionary
            if not isinstance(current_node, dict):
                # Fallback to random action if node is malformed
                return random.randint(0, 2)

            # Check node type
            node_type = current_node.get('type')

            # If it's an action node, return the action
            if node_type == 'action':
                action = current_node.get('value', random.randint(0, 2))
                break

            # If it's a decision node, navigate based on the condition
            if node_type == 'decision':
                # Safely get feature and threshold
                feature = current_node.get('feature', 'agent_hp')
                threshold = current_node.get('threshold', 0)

                # Safely get feature value
                feature_value = state_dict.get(feature, 0)

                # Choose next node based on condition
                if feature_value <= threshold:
                    current_node = current_node.get('left', {})
                else:
                    current_node = current_node.get('right', {})
            else:
                # Unrecognized node type, fallback to random action
                return random.randint(0, 2)

        # Validate action against valid actions if provided
        if valid_actions is not None:
            # If the selected action is invalid
            if not valid_actions[action]:
                # Find valid actions
                valid_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]

                # If there are valid actions, choose one randomly
                if valid_indices:
                    action = random.choice(valid_indices)
                else:
                    # Fallback to a random action if no actions are valid
                    action = random.randint(0, 2)

        return action

    def generate_random_tree(self, depth=0):
        """Generate a random decision tree for action selection"""
        # Hard stop for recursion
        if depth >= self.max_depth:
            # Leaf node - return a random action
            return {
                'type': 'action',
                'value': random.randint(0, 2)
            }

        # Decide between decision and action node
        if depth == 0 or random.random() < 0.7:  # More likely to be a decision node at lower depths
            # Possible features for decision nodes
            feature_choices = [
                'agent_hp',
                'bandit1_hp',
                'bandit2_hp',
                'agent_potions',
                'weaker_bandit',
                'bandit1_dead',
                'bandit2_dead'
            ]
            feature = random.choice(feature_choices)

            # Determine threshold based on feature
            if feature == 'agent_hp':
                threshold = random.uniform(0, 30)
            elif feature in ['bandit1_hp', 'bandit2_hp']:
                threshold = random.uniform(0, 20)
            elif feature == 'agent_potions':
                threshold = random.uniform(0, 3)
            elif feature == 'weaker_bandit':
                threshold = random.randint(0, 2)
            else:  # Binary features
                threshold = 0.5

            return {
                'type': 'decision',
                'feature': feature,
                'threshold': threshold,
                'left': self.generate_random_tree(depth + 1),
                'right': self.generate_random_tree(depth + 1)
            }
        else:
            # Create an action leaf node
            return {
                'type': 'action',
                'value': random.randint(0, 2)
            }

    def mutate(self, mutation_rate=0.2):
        """
        Mutate the decision tree with a certain probability

        Args:
            mutation_rate: Probability of mutation
        """

        def mutate_node(node, depth=0):
            # Mutate with a certain probability
            if random.random() < mutation_rate:
                # Handling different node types safely
                if node.get('type') == 'decision':
                    # For decision nodes, either change the feature or threshold
                    if random.random() < 0.5:
                        # Change the feature
                        feature_choices = [
                            'agent_hp',
                            'bandit1_hp',
                            'bandit2_hp',
                            'agent_potions',
                            'weaker_bandit',
                            'bandit1_dead',
                            'bandit2_dead'
                        ]
                        node['feature'] = random.choice(feature_choices)

                    # Change the threshold based on feature type
                    if node.get('feature') == 'agent_hp':
                        node['threshold'] = random.uniform(0, 30)
                    elif node.get('feature') in ['bandit1_hp', 'bandit2_hp']:
                        node['threshold'] = random.uniform(0, 20)
                    elif node.get('feature') == 'agent_potions':
                        node['threshold'] = random.uniform(0, 3)
                    elif node.get('feature') == 'weaker_bandit':
                        node['threshold'] = random.randint(0, 2)
                    else:
                        node['threshold'] = 0.5

                    # Potentially mutate subtrees
                    if depth < self.max_depth - 1:
                        # Recursively mutate or replace children
                        if random.random() < mutation_rate:
                            node['left'] = self.generate_random_tree(depth + 1)
                        else:
                            mutate_node(node.get('left', {}), depth + 1)

                        if random.random() < mutation_rate:
                            node['right'] = self.generate_random_tree(depth + 1)
                        else:
                            mutate_node(node.get('right', {}), depth + 1)

                elif node.get('type') == 'action':
                    # Change the action
                    node['value'] = random.randint(0, 2)
                else:
                    # If node type is unrecognized, generate a new random node
                    new_node = self.generate_random_tree(depth)
                    node.update(new_node)

        # Start the mutation process at the root
        mutate_node(self.decision_tree)

        return self

    def mutate(self, mutation_rate=0.2):
        """
        Mutate the decision tree with a certain probability

        Args:
            mutation_rate: Probability of mutation
        """

        def mutate_node(node, depth=0):
            # Mutate with a certain probability
            if random.random() < mutation_rate:
                # Handling different node types safely
                if node.get('type') == 'decision':
                    # For decision nodes, either change the feature or threshold
                    if random.random() < 0.5:
                        # Change the feature
                        feature_choices = [
                            'agent_hp',
                            'bandit1_hp',
                            'bandit2_hp',
                            'agent_potions',
                            'weaker_bandit',
                            'bandit1_dead',
                            'bandit2_dead'
                        ]
                        node['feature'] = random.choice(feature_choices)

                    # Change the threshold based on feature type
                    if node.get('feature') == 'agent_hp':
                        node['threshold'] = random.uniform(0, 30)
                    elif node.get('feature') in ['bandit1_hp', 'bandit2_hp']:
                        node['threshold'] = random.uniform(0, 20)
                    elif node.get('feature') == 'agent_potions':
                        node['threshold'] = random.uniform(0, 3)
                    elif node.get('feature') == 'weaker_bandit':
                        node['threshold'] = random.randint(0, 2)
                    else:
                        node['threshold'] = 0.5

                    # Potentially mutate subtrees
                    if depth < self.max_depth - 1:
                        # Recursively mutate or replace children
                        if random.random() < mutation_rate:
                            node['left'] = self.generate_random_tree(depth + 1)
                        else:
                            mutate_node(node.get('left', {}), depth + 1)

                        if random.random() < mutation_rate:
                            node['right'] = self.generate_random_tree(depth + 1)
                        else:
                            mutate_node(node.get('right', {}), depth + 1)

                elif node.get('type') == 'action':
                    # Change the action
                    node['value'] = random.randint(0, 2)
                else:
                    # If node type is unrecognized, generate a new random node
                    new_node = self.generate_random_tree(depth)
                    node.update(new_node)

        # Start the mutation process at the root
        mutate_node(self.decision_tree)

        return self

    def crossover(self, other):
        """
        Perform crossover with another individual

        Args:
            other: Another GeneticPolicyIndividual

        Returns:
            offspring: A new individual created by crossover
        """
        # Create a copy of self
        offspring = GeneticPolicyIndividual(max_depth=self.max_depth)
        offspring.decision_tree = copy.deepcopy(self.decision_tree)

        # Helper function to find all nodes in a tree
        def get_all_nodes(node, node_list, is_decision=True):
            if not isinstance(node, dict):
                return node_list

            node_list.append((node, is_decision))

            if node.get('type') == 'decision':
                # Safely get children, using random tree if not present
                left_node = node.get('left', self.generate_random_tree(depth=1))
                right_node = node.get('right', self.generate_random_tree(depth=1))

                get_all_nodes(left_node, node_list, False)
                get_all_nodes(right_node, node_list, False)

            return node_list

        # Get all nodes from both trees
        self_nodes = []
        other_nodes = []
        get_all_nodes(offspring.decision_tree, self_nodes)
        get_all_nodes(other.decision_tree, other_nodes)

        # Filter out nodes that can be replaced (non-root nodes)
        self_replaceable = [node for node, is_decision in self_nodes if not is_decision]
        other_replaceable = [node for node, is_decision in other_nodes if not is_decision]

        if self_replaceable and other_replaceable:
            # Choose a random node from self to replace
            node_to_replace = random.choice(self_replaceable)
            # Choose a random node from other to use as replacement
            replacement_node = random.choice(other_replaceable)

            # Replace the node safely
            for key in list(node_to_replace.keys()):
                if key in replacement_node:
                    node_to_replace[key] = copy.deepcopy(replacement_node[key])

        return offspring

        def get_action(self, state_dict, valid_actions=None):
            """
            Get action from the decision tree based on the current state

            Args:
                state_dict: Dictionary containing state information
                valid_actions: List of valid actions

            Returns:
                action: Selected action
            """

        # If we have valid actions information, mask invalid actions
        if valid_actions is not None and sum(valid_actions) == 0:
            # No valid actions, return random
            return random.randint(0, 2)

        # Safely navigate through decision tree
        current_node = self.decision_tree

        while True:
            # Ensure node is a valid dictionary
            if not isinstance(current_node, dict):
                # Fallback to random action if node is malformed
                return random.randint(0, 2)

            # Check node type
            node_type = current_node.get('type')

            # If it's an action node, return the action
            if node_type == 'action':
                action = current_node.get('value', random.randint(0, 2))
                break

            # If it's a decision node, navigate based on the condition
            if node_type == 'decision':
                # Safely get feature and threshold
                feature = current_node.get('feature', 'agent_hp')
                threshold = current_node.get('threshold', 0)

                # Safely get feature value
                feature_value = state_dict.get(feature, 0)

                # Choose next node based on condition
                if feature_value <= threshold:
                    current_node = current_node.get('left', {})
                else:
                    current_node = current_node.get('right', {})
            else:
                # Unrecognized node type, fallback to random action
                return random.randint(0, 2)

        # Validate action against valid actions if provided
        if valid_actions is not None:
            # If the selected action is invalid
            if not valid_actions[action]:
                # Find valid actions
                valid_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]

                # If there are valid actions, choose one randomly
                if valid_indices:
                    action = random.choice(valid_indices)
                else:
                    # Fallback to a random action if no actions are valid
                    action = random.randint(0, 2)

        return action

    def generate_random_tree(self, depth=0):
        """Generate a random decision tree for action selection"""
        # Hard stop for recursion
        if depth >= self.max_depth:
            # Leaf node - return a random action
            return {
                'type': 'action',
                'value': random.randint(0, 2)
            }

        # Decide between decision and action node
        if depth == 0 or random.random() < 0.7:  # More likely to be a decision node at lower depths
            # Possible features for decision nodes
            feature_choices = [
                'agent_hp',
                'bandit1_hp',
                'bandit2_hp',
                'agent_potions',
                'weaker_bandit',
                'bandit1_dead',
                'bandit2_dead'
            ]
            feature = random.choice(feature_choices)

            # Determine threshold based on feature
            if feature == 'agent_hp':
                threshold = random.uniform(0, 30)
            elif feature in ['bandit1_hp', 'bandit2_hp']:
                threshold = random.uniform(0, 20)
            elif feature == 'agent_potions':
                threshold = random.uniform(0, 3)
            elif feature == 'weaker_bandit':
                threshold = random.randint(0, 2)
            else:  # Binary features
                threshold = 0.5

            return {
                'type': 'decision',
                'feature': feature,
                'threshold': threshold,
                'left': self.generate_random_tree(depth + 1),
                'right': self.generate_random_tree(depth + 1)
            }
        else:
            # Create an action leaf node
            return {
                'type': 'action',
                'value': random.randint(0, 2)
            }

    def mutate(self, mutation_rate=0.2):
        """
        Mutate the decision tree with a certain probability

        Args:
            mutation_rate: Probability of mutation
        """

        def mutate_node(node, depth=0):
            # Mutate with a certain probability
            if random.random() < mutation_rate:
                # Handling different node types safely
                if node.get('type') == 'decision':
                    # For decision nodes, either change the feature or threshold
                    if random.random() < 0.5:
                        # Change the feature
                        feature_choices = [
                            'agent_hp',
                            'bandit1_hp',
                            'bandit2_hp',
                            'agent_potions',
                            'weaker_bandit',
                            'bandit1_dead',
                            'bandit2_dead'
                        ]
                        node['feature'] = random.choice(feature_choices)

                    # Change the threshold based on feature type
                    if node.get('feature') == 'agent_hp':
                        node['threshold'] = random.uniform(0, 30)
                    elif node.get('feature') in ['bandit1_hp', 'bandit2_hp']:
                        node['threshold'] = random.uniform(0, 20)
                    elif node.get('feature') == 'agent_potions':
                        node['threshold'] = random.uniform(0, 3)
                    elif node.get('feature') == 'weaker_bandit':
                        node['threshold'] = random.randint(0, 2)
                    else:
                        node['threshold'] = 0.5

                    # Potentially mutate subtrees
                    if depth < self.max_depth - 1:
                        # Recursively mutate or replace children
                        if random.random() < mutation_rate:
                            node['left'] = self.generate_random_tree(depth + 1)
                        else:
                            mutate_node(node.get('left', {}), depth + 1)

                        if random.random() < mutation_rate:
                            node['right'] = self.generate_random_tree(depth + 1)
                        else:
                            mutate_node(node.get('right', {}), depth + 1)

                elif node.get('type') == 'action':
                    # Change the action
                    node['value'] = random.randint(0, 2)
                else:
                    # If node type is unrecognized, generate a new random node
                    new_node = self.generate_random_tree(depth)
                    node.update(new_node)

        # Start the mutation process at the root
        mutate_node(self.decision_tree)

        return self

    def save(self, filepath):
        """Save the policy to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.decision_tree, f)

    def load(self, filepath):
        """Load the policy from a file"""
        with open(filepath, 'rb') as f:
            self.decision_tree = pickle.load(f)


class GeneticPolicyTrainer:
    """
    Trainer for genetic algorithm-based policies
    """

    def __init__(self,
                 population_size=50,
                 generations=200,
                 elite_size=5,
                 tournament_size=5,
                 crossover_prob=0.8,
                 mutation_rate=0.2,
                 max_depth=4):

        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.max_depth = max_depth

        # Initialize the population
        self.population = [GeneticPolicyIndividual(max_depth=max_depth) for _ in range(population_size)]

        # Tracking metrics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual_history = []

    def evaluate_population(self, env, episodes_per_individual=5):
        """
        Evaluate the fitness of each individual in the population

        Args:
            env: Battle environment
            episodes_per_individual: Number of episodes to evaluate each individual
        """
        for individual in tqdm(self.population, desc="Evaluating population"):
            total_rewards = 0
            individual.wins = 0
            individual.total_games = episodes_per_individual

            for _ in range(episodes_per_individual):
                # Reset environment
                obs, _ = env.reset()
                done = False
                episode_reward = 0

                while not done:
                    # Extract state information
                    state_dict = self.obs_to_state_dict(obs)

                    # Get valid actions
                    valid_actions = obs[9:12].astype(bool)

                    # Get action from the individual's policy
                    action = individual.get_action(state_dict, valid_actions)

                    # Take action
                    next_obs, reward, done, truncated, _ = env.step(action)

                    # Update state and reward
                    obs = next_obs
                    episode_reward += reward

                # Track results
                total_rewards += episode_reward
                if env.agent_hp > 0:  # Agent won
                    individual.wins += 1

            # Calculate fitness
            individual.fitness = total_rewards / episodes_per_individual
            # Bonus for wins
            individual.fitness += individual.wins * 10

    def obs_to_state_dict(self, obs):
        """
        Convert observation array to a dictionary for easier feature access

        Args:
            obs: Observation array from environment

        Returns:
            state_dict: Dictionary with named features
        """
        agent_hp = obs[0]
        bandit1_hp = obs[1]
        bandit2_hp = obs[2]
        agent_potions = obs[3]

        # Determine which bandit is weaker
        weaker_bandit = 0  # Default: both equal or dead
        if bandit1_hp > 0 and bandit2_hp > 0:
            if bandit1_hp < bandit2_hp:
                weaker_bandit = 1  # Bandit 1 is weaker
            else:
                weaker_bandit = 2  # Bandit 2 is weaker

        # Check if bandits are dead
        bandit1_dead = 1 if bandit1_hp <= 0 else 0
        bandit2_dead = 1 if bandit2_hp <= 0 else 0

        return {
            'agent_hp': agent_hp,
            'bandit1_hp': bandit1_hp,
            'bandit2_hp': bandit2_hp,
            'agent_potions': agent_potions,
            'weaker_bandit': weaker_bandit,
            'bandit1_dead': bandit1_dead,
            'bandit2_dead': bandit2_dead
        }

    def select_parent(self):
        """
        Select a parent using tournament selection

        Returns:
            parent: Selected individual
        """
        # Randomly select individuals for the tournament
        tournament = random.sample(self.population, self.tournament_size)

        # Return the one with the highest fitness
        return max(tournament, key=lambda ind: ind.fitness)

    def create_next_generation(self):
        """
        Create the next generation through selection, crossover, and mutation
        """
        # Sort population by fitness (descending)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Keep track of the best individual
        best_individual = copy.deepcopy(self.population[0])
        self.best_individual_history.append(best_individual)

        # Record fitness metrics
        best_fitness = self.population[0].fitness
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        # Start with the elites
        next_generation = copy.deepcopy(self.population[:self.elite_size])

        # Fill the rest of the population
        while len(next_generation) < self.population_size:
            # Select parents
            parent1 = self.select_parent()

            if random.random() < self.crossover_prob:
                # Crossover
                parent2 = self.select_parent()
                offspring = parent1.crossover(parent2)
            else:
                # No crossover, just copy
                offspring = copy.deepcopy(parent1)

            # Mutation
            offspring.mutate(self.mutation_rate)

            # Add to next generation
            next_generation.append(offspring)

        # Replace the current population
        self.population = next_generation

    def train(self, env, episodes_per_individual=5, save_interval=10):
        """
        Train the population over multiple generations

        Args:
            env: Battle environment
            episodes_per_individual: Number of episodes to evaluate each individual
            save_interval: How often to save the best individual (in generations)
        """
        # Create log file
        with open('genetic_training_log.txt', 'w') as log_file:
            log_file.write("Generation,BestFitness,AvgFitness,BestWinRate\n")

            for generation in range(self.generations):
                print(f"\nGeneration {generation + 1}/{self.generations}")

                # Evaluate the population
                self.evaluate_population(env, episodes_per_individual)

                # Get the best individual
                best_individual = max(self.population, key=lambda ind: ind.fitness)
                best_win_rate = (best_individual.wins / best_individual.total_games) * 100

                # Log and display progress
                print(f"Best fitness: {best_individual.fitness:.2f}")
                print(f"Average fitness: {sum(ind.fitness for ind in self.population) / len(self.population):.2f}")
                print(f"Best win rate: {best_win_rate:.2f}%")

                # Write to log
                log_file.write(
                    f"{generation + 1},{best_individual.fitness:.2f},{sum(ind.fitness for ind in self.population) / len(self.population):.2f},{best_win_rate:.2f}\n")

                # Save the best individual at intervals
                if (generation + 1) % save_interval == 0 or generation == self.generations - 1:
                    best_individual.save(f"genetic_rpg_model_gen{generation + 1}.pkl")
                    print(f"Saved best individual from generation {generation + 1}")

                # Create the next generation (except for the last one)
                if generation < self.generations - 1:
                    self.create_next_generation()

        # Plot training metrics
        self.plot_training_metrics()

        # Return the best individual
        return max(self.population, key=lambda ind: ind.fitness)

    def plot_training_metrics(self):
        """Plot fitness metrics over generations"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history, label='Best Fitness')
        plt.plot(self.avg_fitness_history, label='Avg Fitness')
        plt.title('Fitness over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)

        # Read the log to get win rates
        win_rates = []
        with open('genetic_training_log.txt', 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                win_rates.append(float(parts[3]))

        plt.subplot(1, 2, 2)
        plt.plot(win_rates, label='Best Win Rate')
        plt.title('Win Rate over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Win Rate (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('genetic_training_metrics.png')
        plt.show()


def train_genetic_agent(agent_strength=10, bandit_strength=6,
                        population_size=50, generations=200,
                        episodes_per_individual=5, save_interval=10):
    """
    Train a genetic algorithm-based agent

    Args:
        agent_strength: Strength parameter for the agent
        bandit_strength: Strength parameter for the bandits
        population_size: Size of the genetic algorithm population
        generations: Number of generations to evolve
        episodes_per_individual: Number of episodes to evaluate each individual
        save_interval: How often to save the best individual (in generations)
    """
    # Create environment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)

    # Create trainer
    trainer = GeneticPolicyTrainer(
        population_size=population_size,
        generations=generations,
        elite_size=5,
        tournament_size=5,
        crossover_prob=0.8,
        mutation_rate=0.2,
        max_depth=4
    )

    # Train
    best_individual = trainer.train(env, episodes_per_individual, save_interval)

    # Save the best individual
    best_individual.save("genetic_rpg_model_best.pkl")
    print("Best individual saved as 'genetic_rpg_model_best.pkl'")

    return best_individual


def test_genetic_agent(num_episodes=5, agent_strength=10, bandit_strength=6,
                       model_path="genetic_rpg_model_best.pkl"):
    """
    Test a trained genetic algorithm agent

    Args:
        num_episodes: Number of episodes to test
        agent_strength: Strength parameter for the agent
        bandit_strength: Strength parameter for the bandits
        model_path: Path to the saved model
    """
    # Create environment
    env = BattleEnv(agent_strength=agent_strength, bandit_strength=bandit_strength)

    # Create individual and load model
    individual = GeneticPolicyIndividual()
    individual.load(model_path)

    # Testing metrics
    wins = 0
    total_rewards = 0

    print(f"\nTesting genetic agent over {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Reset environment
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        done = False

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not done:
            # Extract state information
            state_dict = {
                'agent_hp': obs[0],
                'bandit1_hp': obs[1],
                'bandit2_hp': obs[2],
                'agent_potions': obs[3],
                'weaker_bandit': 1 if obs[1] < obs[2] and obs[1] > 0 and obs[2] > 0 else (
                    2 if obs[2] < obs[1] and obs[1] > 0 and obs[2] > 0 else 0),
                'bandit1_dead': 1 if obs[1] <= 0 else 0,
                'bandit2_dead': 1 if obs[2] <= 0 else 0
            }

            # Get valid actions
            valid_actions = obs[9:12].astype(bool)

            # Select action
            action = individual.get_action(state_dict, valid_actions)

            # Print current state and action
            print(f"\nStep {step_count + 1}")
            print(f"Agent HP: {obs[0]:.1f}, Bandit1 HP: {obs[1]:.1f}, Bandit2 HP: {obs[2]:.1f}")
            print(f"Agent Potions: {obs[3]:.1f}, Bandit1 Potions: {obs[4]:.1f}, Bandit2 Potions: {obs[5]:.1f}")
            print(f"Action taken: {['Attack Bandit1', 'Attack Bandit2', 'Use Potion'][action]}")

            # Take action
            next_obs, reward, done, truncated, _ = env.step(action)

            # Update for next step
            obs = next_obs
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


# !/usr/bin/env python3

import argparse
import sys
import os

from genetic_rpg_rl import (
    train_genetic_agent,
    test_genetic_agent,
    GeneticPolicyIndividual
)


def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm RPG Agent Trainer and Tester")

    # Add arguments
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'visualize'],
                        default='train', help='Mode of operation')

    # Training arguments
    parser.add_argument('--agent_strength', type=int, default=10,
                        help='Strength of the agent')
    parser.add_argument('--bandit_strength', type=int, default=6,
                        help='Strength of the bandits')
    parser.add_argument('--population_size', type=int, default=50,
                        help='Size of the genetic population')
    parser.add_argument('--generations', type=int, default=200,
                        help='Number of generations to evolve')
    parser.add_argument('--episodes_per_individual', type=int, default=5,
                        help='Number of episodes to evaluate each individual')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='How often to save the best individual')

    # Testing arguments
    parser.add_argument('--test_episodes', type=int, default=5,
                        help='Number of episodes to test')
    parser.add_argument('--model_path', type=str, default='genetic_rpg_model_best.pkl',
                        help='Path to the trained model')

    # Visualization argument
    parser.add_argument('--vis_episodes', type=int, default=2,
                        help='Number of episodes to visualize')

    # Parse arguments
    args = parser.parse_args()

    # Perform the selected mode
    if args.mode == 'train':
        print("Starting Genetic Algorithm Training...")
        train_genetic_agent(
            agent_strength=args.agent_strength,
            bandit_strength=args.bandit_strength,
            population_size=args.population_size,
            generations=args.generations,
            episodes_per_individual=args.episodes_per_individual,
            save_interval=args.save_interval
        )
        print("Training complete.")

    elif args.mode == 'test':
        print("Starting Model Testing...")
        # Check if model exists before testing
        if not os.path.exists(args.model_path):
            print(f"Error: Model file '{args.model_path}' not found.")
            print("Please train a model first or specify a valid model path.")
            sys.exit(1)

        test_genetic_agent(
            num_episodes=args.test_episodes,
            agent_strength=args.agent_strength,
            bandit_strength=args.bandit_strength,
            model_path=args.model_path
        )

    elif args.mode == 'visualize':
        try:
            from graphic_visualizer import test_visualizer
            print("Starting Model Visualization...")

            # Check if model exists before visualizing
            if not os.path.exists(args.model_path):
                print(f"Error: Model file '{args.model_path}' not found.")
                print("Please train a model first or specify a valid model path.")
                sys.exit(1)

            # Create a policy individual and load the model
            individual = GeneticPolicyIndividual()
            individual.load(args.model_path)

            # Use the generic test_visualizer from graphic_visualizer
            from graphic_visualizer import GameVisualizer

            # Visualize the genetic policy
            visualizer = GameVisualizer()
            visualizer.test_visualizer_gp(num_episodes=5, agent_strength=10, bandit_strength=6,
                                          model_path="genetic_rpg_model_best.pkl")

        except ImportError:
            print("Visualization requires graphic_visualizer.py. Please ensure it's in the same directory.")
        except Exception as e:
            print(f"Visualization error: {e}")


if __name__ == "__main__":
    main()
