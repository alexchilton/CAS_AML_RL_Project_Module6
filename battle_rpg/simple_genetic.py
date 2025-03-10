#!/usr/bin/env python3

import sys
import numpy as np
import random
import pickle
from graphic_env import BattleEnv


class SimpleGeneticPolicy:
    def __init__(self):
        # Simple rule-based policy as a starting point
        self.rules = [
            # Format: [(condition function, condition threshold), action]
            [(lambda state: state['agent_hp'], 10.0), 2],  # If HP < 10, use potion
            [(lambda state: state['bandit1_hp'] < state['bandit2_hp'], 0.5), 0],  # If bandit1 weaker, attack bandit1
            [(lambda state: state['bandit2_hp'] < state['bandit1_hp'], 0.5), 1],  # If bandit2 weaker, attack bandit2
            [(lambda state: state['bandit1_hp'] <= 0, 0.5), 1],  # If bandit1 dead, attack bandit2
            [(lambda state: state['bandit2_hp'] <= 0, 0.5), 0],  # If bandit2 dead, attack bandit1
            [(lambda state: True, 0), 0]  # Default action: attack bandit1
        ]

    def get_action(self, state_dict, valid_actions):
        # Find the first rule that applies
        for (condition_func, threshold), action in self.rules:
            if condition_func(state_dict) < threshold:
                # Check if the action is valid
                if valid_actions[action]:
                    return action

        # If no rule applies or the selected action is invalid, choose a random valid action
        valid_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        if valid_indices:
            return random.choice(valid_indices)
        return 0  # Default to attack bandit1 if nothing else works


def test_genetic_policy():
    """Simple test of genetic policy"""
    env = BattleEnv(agent_strength=10, bandit_strength=6)
    policy = SimpleGeneticPolicy()

    num_episodes = 10
    wins = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            # Convert observation to state dict
            state_dict = {
                'agent_hp': obs[0],
                'bandit1_hp': obs[1],
                'bandit2_hp': obs[2],
                'agent_potions': obs[3]
            }

            # Get valid actions
            valid_actions = obs[9:12].astype(bool)

            # Get action from policy
            action = policy.get_action(state_dict, valid_actions)

            # Take action
            next_obs, reward, done, truncated, _ = env.step(action)

            episode_reward += reward
            step += 1
            obs = next_obs

        if env.agent_hp > 0:  # Agent won
            wins += 1
            result = "WIN"
        else:
            result = "LOSS"

        print(f"Episode {episode + 1}: {result} with reward {episode_reward:.1f} in {step} steps")

    win_rate = (wins / num_episodes) * 100
    print(f"\nWin rate: {win_rate:.1f}% ({wins}/{num_episodes})")


def visualize_genetic_policy(num_episodes=2):
    """Visualize the genetic policy using the game visualizer"""
    try:
        from graphic_visualizer import GameVisualizer
        import pygame

        # Create policy
        policy = SimpleGeneticPolicy()

        # Create visualizer
        visualizer = GameVisualizer(agent_strength=10, bandit_strength=6)

        # Run visualization
        print("\nStarting battle visualization with genetic policy...")
        print("(Close the pygame window to stop)")

        running = True
        episode_running = True
        current_episode = 0
        step_counter = 0

        while running and current_episode < num_episodes:
            visualizer.clock.tick(visualizer.fps)

            # Check if we need to start a new episode
            if not episode_running:
                visualizer.reset_battle()
                episode_running = True
                step_counter = 0
                current_episode += 1
                print(f"\nStarting Episode {current_episode}/{num_episodes}")

            # Draw game
            visualizer.screen.blit(visualizer.background_image, (0, 0))
            visualizer.draw_panel()

            # Update and draw fighters
            visualizer.knight.update()
            visualizer.knight.draw(visualizer.screen)
            for bandit in visualizer.bandit_list:
                bandit.update()
                bandit.draw(visualizer.screen)

            # Update and draw health bars
            visualizer.knight_health_bar.draw(visualizer.knight.hp, visualizer.screen)
            visualizer.bandit1_health_bar.draw(visualizer.bandit1.hp, visualizer.screen)
            visualizer.bandit2_health_bar.draw(visualizer.bandit2.hp, visualizer.screen)

            # Update damage text
            visualizer.damage_text_group.update()
            visualizer.damage_text_group.draw(visualizer.screen)

            # Show episode info
            visualizer.draw_text(f"Episode: {current_episode}/{num_episodes}", visualizer.font, visualizer.blue, 10, 10)
            visualizer.draw_text(f"Step: {step_counter}", visualizer.font, visualizer.blue, 10, 30)

            if visualizer.game_over == 0:
                # Agent turn
                if visualizer.current_fighter == 1:
                    visualizer.action_cooldown += 1
                    if visualizer.action_cooldown >= visualizer.action_wait_time:
                        # Get state
                        state = visualizer.get_state()

                        # Create state dict
                        state_dict = {
                            'agent_hp': state[0],
                            'bandit1_hp': state[1],
                            'bandit2_hp': state[2],
                            'agent_potions': state[3]
                        }

                        # Get valid actions
                        valid_actions = state[9:12].astype(bool)

                        # Get action
                        action = policy.get_action(state_dict, valid_actions)

                        # Execute action
                        if visualizer.execute_agent_action(action):
                            visualizer.current_fighter += 1
                            visualizer.action_cooldown = 0
                            step_counter += 1

                # Enemy turns
                else:
                    visualizer.action_cooldown += 1
                    if visualizer.action_cooldown >= visualizer.action_wait_time:
                        # Get current bandit
                        current_bandit_index = visualizer.current_fighter - 2
                        current_bandit = visualizer.bandit_list[current_bandit_index]

                        # Execute bandit action
                        visualizer.execute_bandit_action(current_bandit, current_bandit_index)

                        visualizer.current_fighter += 1
                        visualizer.action_cooldown = 0

                # Reset fighters cycle
                if visualizer.current_fighter > visualizer.total_fighters:
                    visualizer.current_fighter = 1

                # Check for game over
                alive_bandits = sum(1 for bandit in visualizer.bandit_list if bandit.alive)
                if alive_bandits == 0:
                    visualizer.game_over = 1
                    print(f"Episode {current_episode} - VICTORY! Knight HP: {visualizer.knight.hp}")
                elif not visualizer.knight.alive:
                    visualizer.game_over = -1
                    print(f"Episode {current_episode} - DEFEAT!")

            # Display game over messages
            if visualizer.game_over != 0:
                if visualizer.game_over == 1:
                    visualizer.screen.blit(visualizer.victory_image, (250, 50))
                    visualizer.draw_text("VICTORY!", visualizer.title_font, visualizer.green, 340, 150)
                else:
                    visualizer.screen.blit(visualizer.defeat_image, (290, 50))
                    visualizer.draw_text("DEFEAT!", visualizer.title_font, visualizer.red, 350, 150)

                # Display continue button
                pygame.draw.rect(visualizer.screen, visualizer.blue, visualizer.restart_button)
                visualizer.draw_text("Continue", visualizer.font, (255, 255, 255), 360, 365)

                # After a short delay, move to the next episode
                visualizer.action_cooldown += 1
                if visualizer.action_cooldown >= visualizer.action_wait_time * 2:
                    episode_running = False

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    episode_running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if restart button was clicked
                    if visualizer.game_over != 0 and visualizer.restart_button.collidepoint(event.pos):
                        episode_running = False

            pygame.display.update()

        pygame.quit()

    except ImportError as e:
        print(f"Could not import visualization module: {e}")
        print("Make sure graphic_visualizer.py is in the same directory.")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        visualize_genetic_policy()
    else:
        test_genetic_policy()