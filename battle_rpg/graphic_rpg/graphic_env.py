import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class BattleEnv(gym.Env):
    def __init__(self, strenght):
        # super().__init__(current_fighter, total_fighters, action_cooldown, action_wait_time, attack, potion, potion_effect, game_over )
        super().__init__()

        # Actions: attck bandit 1, attack bandit 2, use potion --> to be updated with what we want
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [agent_hp (knight), bandit_hp (times n of bandits), no. potions agent, no. potions enemies (times no. of enemies)]
        # 1 + 1*2 + 1 + 1*2 = 6
        self.observation_space = spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32)
        
        # Game parameters
        # current_fighter = 1
        # total_fighters = 3
        # action_cooldown = 0
        # action_wait_time = 90
        # attack = False
        # potion = False
        # potion_effect = 15
        # game_over=0

        self.total_fighters = 3
        self.max_hp = 30
        self.max_potions = 3
        self.enemy_max_hp = 20
        self.max_enemy_potions = 2
        self.strenght= strenght

        # calculation of damage with randomness
        rand = random.randint(-5, 5)
        damage = self.strenght + rand
        self.attack_damage = damage
        self.potion_effect = 15

        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Observation space: [agent_hp (knight), bandit_hp (times n of bandits), no. potions agent, no. potions enemies (times no. of enemies)]
        self.agent_hp = self.max_hp
        self.bandit1_hp = self.enemy_max_hp
        self.bandit2_hp = self.enemy_max_hp
        self.agent_potions = self.max_potions
        self.bandit1_potions = self.max_enemy_potions
        self.bandit2_potions = self.max_enemy_potions

        # reset last action
        self.last_action_agent = -1
        self.last_action_bandit1 = -1
        self.last_action_bandit2 = -1

        return self._get_state(), {}
    
    def _get_state(self):
        # Observation space: [agent_hp (knight), bandit_hp (times n of bandits), no. potions agent, no. potions enemies (times no. of enemies)]
        return np.array([
            self.agent_hp,
            self.bandit1_hp,
            self.bandit2_hp,
            self.agent_potions,
            self.bandit1_potions,
            self.bandit2_potions, 
            self.last_action_agent,
            self.last_action_bandit1,
            self.last_action_bandit2
        ], dtype=np.float32)
    
    # def _bandit1_action(self):
    #    # check if bandit needs to heal
    #     if (self.bandit1_hp / self.enemy_max_hp) < 0.5 and self.bandit1_potions > 0:
    #         return np.random.choice([1,2], p=[0.2, 0.8])  # if the hp are low more likely to heal
    #     return np.random.choice([1,2], p=[0.8, 0.2])  # otherwise attack 

    # def _bandit2_action(self):
    #    # check if bandit needs to heal
    #     if (self.bandit2_hp / self.enemy_max_hp) < 0.5 and self.bandit2_potions > 0:
    #         return np.random.choice([1,2], p=[0.2, 0.8])  # if the hp are low more likely to heal
    #     return np.random.choice([1,2], p=[0.8, 0.2])  # otherwise attack  

    def _bandit_action(self, bandit_hp, bandit_potions):
        # Check if bandit needs to heal
        if (bandit_hp / self.enemy_max_hp) < 0.5 and bandit_potions > 0:
            return np.random.choice([0, 1], p=[0.2, 0.8])  # More likely to heal when low HP
        return np.random.choice([0, 1], p=[0.8, 0.2])  # More likely to attack otherwise


    def step(self, action):
        # initialize the reward
        reward = 0

        # action recording
        self.last_action_agent = action
        bandit1_action = self._bandit_action(self.bandit1_hp, self.bandit1_potions)
        self.last_bandit1_action = bandit1_action
        bandit2_action = self._bandit_action(self.bandit2_hp, self.bandit2_potions)
        self.last_bandit2_action = bandit2_action
        
        # Process actions
        agent_damage_to_bandit1 = 0
        agent_damage_to_bandit2 = 0
        bandit1_damage = 0
        bandit2_damage = 0

        
        # Agent action
        if action == 0:  # Attack bandit 1
            agent_damage_to_bandit1 = self.attack_damage 
            reward += 1
        elif action == 1: # Attack bandit 2
            agent_damage_to_bandit2 = self.attack_damage
            reward += 1
        elif action == 2:  # heal
            if self.agent_potions > 0:
                if self.max_hp - self.agent_hp > self.potion_effect:
                    heal_amount = self.potion_effect
                    reward += 1
                else:
                    heal_amount = self.max_hp - self.agent_hp
                    reward += 0.5
                self.agent_hp += heal_amount
                self.agent_potions -= 1
        
        # Bandit1 action
        if bandit1_action == 0:
            bandit1_damage = self.attack_damage
        else:  # use the potion
            if self.bandit1_potions > 0:
                heal_amount = min(self.potion_effect, self.enemy_max_hp - self.bandit1_hp)
                self.bandit1_hp += heal_amount
                self.bandit1_potions -= 1
        # Bandit2 action
        if bandit2_action == 0:  # Attack
            bandit2_damage = self.attack_damage
        else:  # Use potion
            if self.bandit2_potions > 0:
                heal_amount = min(self.potion_effect, self.enemy_max_hp - self.bandit2_hp)
                self.bandit2_hp += heal_amount
                self.bandit2_potions -= 1
        
        # Apply damage
        self.bandit1_hp -= agent_damage_to_bandit1
        self.bandit2_hp -= agent_damage_to_bandit2
        self.agent_hp -= (bandit1_damage + bandit2_damage)

        # ensure that hp values don't go below 0
        self.agent_hp = max(0, self.agent_hp)
        self.bandit1_hp = max(0, self.bandit1_hp)
        self.bandit2_hp = max(0, self.bandit2_hp)

        # Check game over
        done = (self.agent_hp <= 0) or (self.bandit1_hp <= 0 and self.bandit2_hp <= 0)
        truncated = False
        
        # Calculate reward
        if done:
            if self.agent_hp <= 0:  # Agent lost
                reward = -10
            else:  # Agent won
                reward = 10 + self.agent_hp  # Bonus for remaining HP
        
        return self._get_state(), reward, done, truncated, {}