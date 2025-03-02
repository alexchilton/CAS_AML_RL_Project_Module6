import pygame
import random
import numpy as np
from stable_baselines3 import PPO

class Fighter():
    def __init__(self, x, y, name, max_hp, strenght, potions):
        self.name = name
        self.max_hp = max_hp
        self.hp = max_hp
        self.strenght = strenght
        self.start_potions = potions
        self.potions = potions
        self.alive = True
        self.animation_list = []
        self.frame_index = 0
        self.action = 0  # 0: Idle , 1: attack, 2: Hurt, 3: Death
        self.update_time = pygame.time.get_ticks()
        
        # load idle images
        temp_list = []
        for i in range(8):
            img = pygame.image.load(f'img/{self.name}/Idle/{i}.png')
            img = pygame.transform.scale(img, (img.get_width()*3, img.get_height()*3))
            temp_list.append(img)
        self.animation_list.append(temp_list)
        
        # load attack images
        temp_list = []
        for i in range(8):
            img = pygame.image.load(f'img/{self.name}/Attack/{i}.png')
            img = pygame.transform.scale(img, (img.get_width()*3, img.get_height()*3))
            temp_list.append(img)
        self.animation_list.append(temp_list)
        
        # load hurt images
        temp_list = []
        for i in range(3):
            img = pygame.image.load(f'img/{self.name}/Hurt/{i}.png')
            img = pygame.transform.scale(img, (img.get_width()*3, img.get_height()*3))
            temp_list.append(img)
        self.animation_list.append(temp_list)
        
        # load death images
        temp_list = []
        for i in range(10):
            img = pygame.image.load(f'img/{self.name}/Death/{i}.png')
            img = pygame.transform.scale(img, (img.get_width()*3, img.get_height()*3))
            temp_list.append(img)
        self.animation_list.append(temp_list)
        
        self.image = self.animation_list[self.action][self.frame_index]
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    def update(self):
        animation_cooldown = 100
        # handle animation
        self.image = self.animation_list[self.action][self.frame_index]
        if pygame.time.get_ticks() - self.update_time > animation_cooldown:
            self.update_time = pygame.time.get_ticks()
            self.frame_index += 1
        if self.frame_index >= len(self.animation_list[self.action]):
            if self.action == 3:
                self.frame_index = len(self.animation_list[self.action]) - 1
            else:            
                self.idle()

    def idle(self):
        self.action = 0
        self.frame_index = 0
        self.update_time = pygame.time.get_ticks()

    def attack(self, target):
        rand = random.randint(-5, 5)
        damage = self.strenght + rand
        target.hp -= damage
        target.hurt()
        if target.hp < 1:
            target.hp = 0
            target.alive = False
            target.death()
        damage_text = DamageText(target.rect.centerx, target.rect.y, str(damage), (255, 0, 0))
        damage_text_group.add(damage_text)
        self.action = 1
        self.frame_index = 0
        self.update_time = pygame.time.get_ticks()

    def hurt(self):
        self.action = 2
        self.frame_index = 0
        self.update_time = pygame.time.get_ticks()

    def death(self):
        self.action = 3
        self.frame_index = 0
        self.update_time = pygame.time.get_ticks()

    def reset(self):
        self.alive = True
        self.potions = self.start_potions
        self.hp = self.max_hp
        self.frame_index = 0
        self.action = 0
        self.update_time = pygame.time.get_ticks()

    def draw(self, surface):
        surface.blit(self.image, self.rect)

class HealthBar():
    def __init__(self, x, y, hp, max_hp):
        self.x = x
        self.y = y
        self.hp = hp
        self.max_hp = max_hp

    def draw(self, hp, surface):
        self.hp = hp
        ratio = self.hp / self.max_hp
        pygame.draw.rect(surface, (255, 0, 0), (self.x, self.y, 150, 15))
        pygame.draw.rect(surface, (0, 255, 0), (self.x, self.y, 150 * ratio, 15))

class DamageText(pygame.sprite.Sprite):
    def __init__(self, x, y, damage, colour):
        pygame.sprite.Sprite.__init__(self)
        self.font = pygame.font.SysFont('Times New Roman', 21)
        self.image = self.font.render(damage, True, colour)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.counter = 0

    def update(self):
        self.rect.y -= 1
        self.counter += 1
        if self.counter > 30:
            self.kill()

class GameVisualizer:
    def __init__(self):
        pygame.init()
        
        # Game window settings
        self.bottom_panel = 150
        self.screen_width = 800
        self.screen_height = 400 + self.bottom_panel
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Battle Visualization')
        
        # Game settings
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.current_fighter = 1
        self.total_fighters = 3
        self.action_cooldown = 0
        self.action_wait_time = 90
        self.game_over = 0
        self.potion_effect = 15
        
        # Colors
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.powderblue = (176, 224, 230)
        
        # Load images and setup UI
        self.setup_graphics()
        
        # Initialize sprites
        global damage_text_group
        damage_text_group = pygame.sprite.Group()
        
        # Initialize fighters
        self.setup_fighters()
        
        # Health bars
        self.setup_health_bars()
        
        # Load the trained model
        self.model = PPO.load("graphic_rpg_model")

    def setup_graphics(self):
        # Load all images
        self.background_image = pygame.image.load('img/Background/background.jpg').convert_alpha()
        self.panel_image = pygame.image.load('img/Icons/panel_double.png').convert_alpha()
        self.victory_image = pygame.image.load('img/Icons/victory.png').convert_alpha()
        self.defeat_image = pygame.image.load('img/Icons/defeat.png').convert_alpha()
        self.font = pygame.font.SysFont('Times New Roman', 21)

    def setup_fighters(self):
        self.knight = Fighter(200, 260, 'Knight', 30, 10, 3)
        self.bandit1 = Fighter(550, 270, 'Bandit', 20, 6, 1)
        self.bandit2 = Fighter(700, 270, 'Bandit', 20, 6, 1)
        self.bandit_list = [self.bandit1, self.bandit2]

    def setup_health_bars(self):
        self.knight_health_bar = HealthBar(120, self.screen_height - self.bottom_panel + 40, 
                                         self.knight.hp, self.knight.max_hp)
        self.bandit1_health_bar = HealthBar(530, self.screen_height - self.bottom_panel + 40, 
                                          self.bandit1.hp, self.bandit1.max_hp)
        self.bandit2_health_bar = HealthBar(530, self.screen_height - self.bottom_panel + 90, 
                                          self.bandit2.hp, self.bandit2.max_hp)

    def draw_text(self, text, font, text_col, x, y):
        img = font.render(text, True, text_col)
        self.screen.blit(img, (x, y))

    def draw_panel(self):
        self.screen.blit(self.panel_image, (0, self.screen_height - self.bottom_panel))
        self.draw_text(f'{self.knight.name} HP: {self.knight.hp}', self.font, self.blue, 
                      120, self.screen_height - self.bottom_panel + 10)
        for count, i in enumerate(self.bandit_list):
            self.draw_text(f'{i.name} HP: {i.hp}', self.font, self.powderblue, 
                          530, (self.screen_height - self.bottom_panel + 10) + count * 50)

    def get_state(self):
        valid_actions = [
        1 if self.bandit1_hp > 0 else 0,  # Attack Bandit 1 valid if bandit1 is alive
        1 if self.bandit2_hp > 0 else 0,  # Attack Bandit 2 valid if bandit2 is alive
        1 if self.agent_potions > 0 else 0  # Use potion valid if agent has potions
    ]
        return np.array([
            self.knight.hp,
            self.bandit1.hp,
            self.bandit2.hp,
            self.knight.potions,
            self.bandit1.potions,
            self.bandit2.potions,
            -1,  # last_action_agent
            -1,  # last_action_bandit1
            -1,   # last_action_bandit2
            *valid_actions, 
            self.agent_potions, 
            self.bandit1_potions, 
            self.bandit2_potions
        ], dtype=np.float32)

    def execute_agent_action(self, action):
        if action == 0:  # Attack bandit 1
            if self.bandit1.alive:
                self.knight.attack(self.bandit1)
                return True
        elif action == 1:  # Attack bandit 2
            if self.bandit2.alive:
                self.knight.attack(self.bandit2)
                return True
        elif action == 2:  # Use potion
            if self.knight.potions > 0:
                if self.knight.max_hp - self.knight.hp > self.potion_effect:
                    heal_amount = self.potion_effect
                else:
                    heal_amount = self.knight.max_hp - self.knight.hp
                self.knight.hp += heal_amount
                self.knight.potions -= 1
                damage_text = DamageText(self.knight.rect.centerx, self.knight.rect.y, 
                                       str(heal_amount), self.green)
                damage_text_group.add(damage_text)
                return True
        return False

    def run_visualization(self, num_episodes=1):
        running = True
        while running:                     
            # Draw game state
            self.clock.tick(self.fps)
            self.screen.blit(self.background_image, (0, 0))
            self.draw_panel()
            
            # Update and draw fighters
            self.knight.update()
            self.knight.draw(self.screen)
            for bandit in self.bandit_list:
                bandit.update()
                bandit.draw(self.screen)
            
            # Update and draw health bars
            self.knight_health_bar.draw(self.knight.hp, self.screen)
            self.bandit1_health_bar.draw(self.bandit1.hp, self.screen)
            self.bandit2_health_bar.draw(self.bandit2.hp, self.screen)
            
            # Update damage text
            damage_text_group.update()
            damage_text_group.draw(self.screen)

            if self.game_over == 0:
                # Agent turn
                if self.current_fighter == 1:
                    self.action_cooldown += 1
                    if self.action_cooldown >= self.action_wait_time:
                        # Get state and predict action
                        state = self.get_state()
                        action, _ = self.model.predict(state, deterministic=True)
                        
                        # Execute action
                        if self.execute_agent_action(action):
                            self.current_fighter += 1
                            self.action_cooldown = 0
                
                # Enemy turns
                else:
                    self.action_cooldown += 1
                    if self.action_cooldown >= self.action_wait_time:
                        # Simple AI for bandits
                        current_bandit = self.bandit_list[self.current_fighter - 2]
                        if current_bandit.alive:
                            if (current_bandit.hp / current_bandit.max_hp) < 0.5 and current_bandit.potions > 0:
                                # Heal
                                heal_amount = min(self.potion_effect, current_bandit.max_hp - current_bandit.hp)
                                current_bandit.hp += heal_amount
                                current_bandit.potions -= 1
                                damage_text = DamageText(current_bandit.rect.centerx, current_bandit.rect.y, 
                                                    str(heal_amount), self.green)
                                damage_text_group.add(damage_text)
                            else:
                                # Attack
                                current_bandit.attack(self.knight)
                        
                        self.current_fighter += 1
                        self.action_cooldown = 0
                
                # Reset fighters
                if self.current_fighter > self.total_fighters:
                    self.current_fighter = 1

            # Check for game over
            alive_bandits = sum(1 for bandit in self.bandit_list if bandit.alive)
            if alive_bandits == 0:
                self.game_over = 1
            elif not self.knight.alive:
                self.game_over = -1

            if self.game_over != 0:
                if self.game_over == 1:
                    self.screen.blit(self.victory_image, (250, 50))
                else:
                    self.screen.blit(self.defeat_image, (290, 50))

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    episode_running = False

            pygame.display.update()

        pygame.quit()               