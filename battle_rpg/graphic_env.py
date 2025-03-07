import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class BattleEnv(gym.Env):
    def __init__(self, agent_strength= 10, bandit_strength = 6):
        # super().__init__(current_fighter, total_fighters, action_cooldown, action_wait_time, attack, potion, potion_effect, game_over )
        super().__init__()

        # Actions: attck bandit 1, attack bandit 2, use potion --> to be updated with what we want
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [agent_hp, bandit1_hp, bandit2_hp, agent_potions, bandit1_potions, bandit2_potions,
        #                    last_action_agent, last_action_bandit1, last_action_bandit2]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0]), 
            high=np.array([100, 100, 100, 100, 100, 100, 2, 2, 2, 1, 1, 1, 3, 1, 1]), 
            dtype=np.float32
        )

        self.total_fighters = 3
        self.current_fighter = 1
        self.max_hp = 30
        self.max_potions = 3
        self.enemy_max_hp = 20
        self.max_enemy_potions = 1
        self.agent_strength= agent_strength
        self.bandit_strength = bandit_strength
        
        # Attack damage to reset at each episode
        self.agent_attack_damage = None 
        self.bandit_attack_damage = None
        self.potion_effect = 15
    

    def _calculate_attack_damage(self):
        agent_rand = random.randint(-5, 5)
        bandit_rand = random.randint(-5, 5)
        self.agent_attack_damage = self.agent_strength + agent_rand
        self.bandit_attack_damage = self.bandit_strength + bandit_rand
        #print(f"[DEBUG] New Damage Roll - Agent: {self.agent_attack_damage}, Bandit: {self.bandit_attack_damage}")
        
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
            # Determine which actions are valid:
    # 1 if valid, 0 otherwise.
        valid_actions = [
        1 if self.bandit1_hp > 0 else 0,  # Attack Bandit 1 valid if bandit1 is alive
        1 if self.bandit2_hp > 0 else 0,  # Attack Bandit 2 valid if bandit2 is alive
        1 if self.agent_potions > 0 else 0  # Use potion valid if agent has potions
    ]
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
            self.last_action_bandit2, 
            *valid_actions, 
            self.agent_potions, 
            self.bandit1_potions, 
            self.bandit2_potions
        ], dtype=np.float32)
    

    def _bandit_action(self, bandit_hp, bandit_potions):
        # Check if bandit needs to heal
        if (bandit_hp / self.enemy_max_hp) < 0.5 and bandit_potions > 0:
            return 1  # Must heal when low HP
        return 0  # Must attack otherwise
    
    def _handle_agent_turn(self, action):
        reward = 0

        if self.agent_hp > 0:
            reward += 0.5   # Small reward for surviving another full turn cycle
        
        # Agent action
        if action == 0:  # Attack bandit 1
            if self.bandit1_hp > 0:
                agent_damage_to_bandit1 = self.agent_attack_damage
                self.bandit1_hp -= agent_damage_to_bandit1
                # reward proportional to damage dealt
                # reward += agent_damage_to_bandit1 * 0.2
                reward += 1
                # check if bandit died AFTER damage is applied
                # if self.bandit1_hp>0:
                #     reward += (1-(self.bandit1_hp/self.enemy_max_hp))*3  # bonus for attacking dying enemies
                    
                if self.bandit1_hp <= 0:
                    self.bandit1_hp = 0  # ensure hp doesn't go negative
                    reward += 10  # previously 5 --> bonus for kill
            else:
                reward = -10 # small penalty for forbidden action, should not do due to mask
                
        elif action == 1:  # Attack bandit 2
            if self.bandit2_hp > 0:
                agent_damage_to_bandit2 = self.agent_attack_damage
                #print(f"Before attack: Bandit2 HP = {self.bandit2_hp}")
                #print(f"Damage dealt = {agent_damage_to_bandit2}")
                self.bandit2_hp -= agent_damage_to_bandit2
                #print(f"After attack: Bandit2 HP = {self.bandit2_hp}")
                #reward += agent_damage_to_bandit2 * 0.2
                reward += 1

                # if self.bandit2_hp > 0:
                #     reward += (1 - (self.bandit2_hp / self.enemy_max_hp)) *3 # bonus for attacking dying enemies
                
                if self.bandit2_hp <= 0:
                    self.bandit2_hp = 0
                    reward += 10  # previously 5
            else:
                reward = -10 # small penalty for forbidden action, should not do due to mask
                
        elif action == 2:  # Heal
            if self.agent_potions > 0:
                if self.max_hp - self.agent_hp > self.potion_effect:
                    heal_amount = self.potion_effect
                else:
                    heal_amount = self.max_hp - self.agent_hp
                
                # Store HP before healing for reward calculation
                hp_percentage_before = self.agent_hp / self.max_hp
                
                self.agent_hp += heal_amount
                self.agent_potions -= 1
                
                # Strategic reward based on when healing occurred
                if hp_percentage_before < 0.3:  # critical hp
                    reward += 4
                elif hp_percentage_before < 0.7:  # medium hp
                    reward += 2
                else:  # high hp
                    reward -= 2  # penalize unnecessary healing
                    
                # if self.bandit1_hp <= 0 or self.bandit2_hp <= 0:
                #     reward -= 1  # Small penalty for healing when one enemy is already dead
                
                # # consider state when healing
                # alive_enemies = (self.bandit1_hp > 0) + (self.bandit2_hp > 0)
                # if alive_enemies == 1:
                #     reward -= 2 # Bigger penalty for healing with only one enemy left
        
        return reward

    # def _handle_agent_turn(self, action):
    #     reward = 0

    #     # Tiny reward for surviving another full turn cycle
    #     if self.agent_hp > 0:
    #         reward += 1   

    #     # ==== ATTACKING BANDIT 1 ====
    #     if action == 0:
    #         if self.bandit1_hp > 0:
    #             damage = self.agent_attack_damage
    #             self.bandit1_hp -= damage
    #             reward += 1  # Base reward for attacking

    #             # **Encourage finishing off weak enemies**
    #             if self.bandit1_hp <= 0:
    #                 self.bandit1_hp = 0
    #                 reward += 10  # Big bonus for eliminating an enemy
    #             elif self.bandit1_hp < 5:  
    #                 reward += 5  # Encourage securing the kill quickly

    #         else:
    #             reward = -10  # Small penalty for attacking a dead enemy (shouldn't happen due to masking)

    #     # ==== ATTACKING BANDIT 2 ====
    #     elif action == 1:
    #         if self.bandit2_hp > 0:
    #             damage = self.agent_attack_damage
    #             self.bandit2_hp -= damage
    #             reward += 1  # Base reward for attacking

    #             # **Encourage finishing off weak enemies**
    #             if self.bandit2_hp <= 0:
    #                 self.bandit2_hp = 0
    #                 reward += 10  # Big bonus for eliminating an enemy
    #             elif self.bandit2_hp < 5:  
    #                 reward += 5  # Encourage securing the kill quickly

    #         else:
    #             reward = -10  # Small penalty for attacking a dead enemy (shouldn't happen due to masking)

    #     # ==== USING POTION ====
    #     elif action == 2:
    #         if self.agent_potions > 0:
    #             heal_amount = min(self.potion_effect, self.max_hp - self.agent_hp)
    #             hp_percentage_before = self.agent_hp / self.max_hp

    #             self.agent_hp += heal_amount
    #             self.agent_potions -= 1

    #             # **Strategic healing reward**
    #             if hp_percentage_before < 0.3:
    #                 reward += 4  # Very low HP, good decision!
    #             elif hp_percentage_before < 0.7:
    #                 reward += 2  # Medium HP, still reasonable
    #             else:
    #                 reward -= 2  # Penalize unnecessary healing at high HP

    #     return reward


    def _handle_bandit1_turn(self):
        reward = 0
        if self.bandit1_hp > 0:  # Only act if alive
            # Get bandit's action (already recorded in step method)
            bandit_action = self.last_bandit1_action
            
            if bandit_action == 0:  # Attack
                self.agent_hp -= self.bandit_attack_damage
                self.agent_hp = max(0, self.agent_hp)  # Ensure HP doesn't go negative
                
            else:  # Heal
                if self.bandit1_potions > 0:
                    heal_amount = min(self.potion_effect, self.enemy_max_hp - self.bandit1_hp)
                    self.bandit1_hp += heal_amount
                    self.bandit1_potions -= 1
        
        return reward

    def _handle_bandit2_turn(self):
        reward = 0
        if self.bandit2_hp > 0:  # Only act if alive
            # Get bandit's action (already recorded in step method)
            bandit_action = self.last_bandit2_action
            
            if bandit_action == 0:  # Attack
                self.agent_hp -= self.bandit_attack_damage
                self.agent_hp = max(0, self.agent_hp)  # Ensure HP doesn't go negative
                
            else:  # Heal
                if self.bandit2_potions > 0:
                    heal_amount = min(self.potion_effect, self.enemy_max_hp - self.bandit2_hp)
                    self.bandit2_hp += heal_amount
                    self.bandit2_potions -= 1
        
        return reward


    def step(self, action):
        # initialize the reward
        reward = 0
        self._calculate_attack_damage()


        # Record action only for current fighter
        if self.current_fighter == 1:
            self.last_action_agent = action
        elif self.current_fighter == 2:
            self.last_bandit1_action = self._bandit_action(self.bandit1_hp, self.bandit1_potions)
        elif self.current_fighter == 3:
            self.last_bandit2_action = self._bandit_action(self.bandit2_hp, self.bandit2_potions)

        # Process only the current fighter's action
        if self.current_fighter == 1:  # Agent's turn
            reward = self._handle_agent_turn(action)
            #print(f"[DEBUG] After Agent Attack - Bandit1 HP: {self.bandit1_hp}, Bandit2 HP: {self.bandit2_hp}")
        elif self.current_fighter == 2:  # Bandit1's turn
            reward = self._handle_bandit1_turn()
        elif self.current_fighter == 3:  # Bandit2's turn
            reward = self._handle_bandit2_turn()
        
        # reward clipping 
        reward = np.clip(reward, -10, 10)

        self.current_fighter += 1
        if self.current_fighter > self.total_fighters:
            self.current_fighter = 1



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
                reward = -30
            else:  # Agent won
                remaining_hp_percentage = self.agent_hp / self.max_hp
                reward += 30 + (remaining_hp_percentage * 5)  # Bonus for remaining HP
        #print(f"[DEBUG] Before Returning Step - Agent HP: {self.agent_hp}, Bandit1 HP: {self.bandit1_hp}, Bandit2 HP: {self.bandit2_hp}")
        return self._get_state(), reward, done, truncated, {}