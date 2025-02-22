import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import numpy as np

class BattleVisualizer:
    def __init__(self, width=800, height=400, human_controlled=False):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.human_controlled=human_controlled
        pygame.display.set_caption("Battle Game Visualization")
        
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        self.font = pygame.font.Font(None, 36)

        # Get boss name at initialization if human-controlled
        if self.human_controlled:
            self.boss_name = input("Choose boss name: ")
        else:
            self.boss_name = "Boss"
        
    def draw_health_bar(self, x, y, width, height, health, max_health):
        pygame.draw.rect(self.screen, self.BLACK, (x, y, width, height), 2)
        health_width = int((float(health) / max_health) * (width - 4))
        if health_width > 0:
            pygame.draw.rect(self.screen, self.GREEN, (x + 2, y + 2, health_width, height - 4))
        
    def draw_action(self, x, y, action):
        actions = ["Attack1", "Attack2", "Defend", "Stance"]
        text = self.font.render(f"Action: {actions[int(action)]}", True, self.BLACK)
        self.screen.blit(text, (x, y))
        
    def visualize_state(self, state, action):
        if isinstance(state, np.ndarray):
            state = state.tolist()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            
        self.screen.fill(self.WHITE)
        
        # Draw health bars
        self.draw_health_bar(50, 50, 300, 30, state[0], 100)
        self.draw_health_bar(450, 50, 300, 30, state[1], 100)
        
        # Draw labels
        agent_text = self.font.render("Agent", True, self.BLACK)
        boss_text = self.font.render(self.boss_name, True, self.BLACK)
        self.screen.blit(agent_text, (170, 20))
        self.screen.blit(boss_text, (570, 20))
        
        # Draw HP values
        agent_hp = self.font.render(f"HP: {state[0]:.1f}", True, self.BLACK)
        boss_hp = self.font.render(f"HP: {state[1]:.1f}", True, self.BLACK)
        self.screen.blit(agent_hp, (170, 90))
        self.screen.blit(boss_hp, (570, 90))
        
        # Draw current action
        self.draw_action(300, 200, action)
        
        # Draw buffs
        if state[4] > 1.0:
            buff_text = self.font.render("BUFFED!", True, self.BLUE)
            self.screen.blit(buff_text, (50, 150))
        if state[5] > 1.0:
            buff_text = self.font.render("BUFFED!", True, self.RED)
            self.screen.blit(buff_text, (450, 150))
            
        pygame.display.flip()
        pygame.time.wait(1000)  
        
    def close(self):
        pygame.quit()