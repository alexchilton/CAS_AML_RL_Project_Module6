import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BattleEnv(gym.Env):
    def __init__(self, human_controlled=False):
        super().__init__()

        self.human_controlled=human_controlled
        
        # Actions: attack1, attack2, defend, stance --> to be updated with what we want
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [agent_hp, boss_hp, last_action_agent, last_action_boss, stance_buff_agent, stance_buff_boss]
        self.observation_space = spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32)
        
        # Game parameters
        self.max_hp = 100
        self.attack1_damage = 15
        self.attack2_damage = 20
        #self.defense_reduction = 0.5
        self.defense_reduction = 0.9
        self.stance_buff = 1.5
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.agent_hp = self.max_hp
        self.boss_hp = self.max_hp
        self.last_action_agent = -1
        self.last_action_boss = -1
        self.stance_buff_agent = 1.0
        self.stance_buff_boss = 1.0
        return self._get_state(), {}
    
    def _get_state(self):
        return np.array([
            self.agent_hp,
            self.boss_hp,
            self.last_action_agent,
            self.last_action_boss,
            self.stance_buff_agent,
            self.stance_buff_boss
        ], dtype=np.float32)
    
    def _boss_action(self):
        # Simple boss AI: randomly choose action with some basic strategy --> we can add option that a person can play too agaist agent
        if self.human_controlled:
            while True:
                try:
                    action=int(input("Choose boss action (1: Attack1, 2: Attack2, 3: Defend, 4: Stance): "))
                    if action in [1,2,3,4]:
                        return action
                    else:
                        print('Invalid action! choose between 0-3')
                except ValueError:
                    print('Invalid input! Please enter a number between 0-3.')
        else: 
            if self.boss_hp < 30:  # Low HP - more likely to defend
                return np.random.choice([1, 2, 3,4], p=[0.3, 0.3, 0.3, 0.1])
            return np.random.choice([1, 2, 3, 4], p=[0.4, 0.4, 0.1, 0.1])
    
    def step(self, action):
        self.last_action_agent = action
        boss_action = self._boss_action()
        self.last_action_boss = boss_action
        
        # Process stance buffs
        agent_damage_mult = self.stance_buff_agent
        boss_damage_mult = self.stance_buff_boss
        self.stance_buff_agent = 1.0
        self.stance_buff_boss = 1.0
        #self.stance_buff_boss = 1.5
        
        # Process actions
        agent_damage = 0
        boss_damage = 0
        agent_defending = False
        boss_defending = False
        
        # Agent action
        if action == 1:  # Attack 1
            agent_damage = self.attack1_damage * agent_damage_mult
        elif action == 2:  # Attack 2
            agent_damage = self.attack2_damage * agent_damage_mult
        elif action == 3:  # Defend
            agent_defending = True
        elif action == 4:  # Stance
            self.stance_buff_agent = self.stance_buff
        
        # Boss action
        if boss_action == 1:
            boss_damage = self.attack1_damage * boss_damage_mult
        elif boss_action == 2:
            boss_damage = self.attack2_damage * boss_damage_mult
        elif boss_action == 3:
            boss_defending = True
        elif boss_action == 4:
            self.stance_buff_boss = self.stance_buff
        
        # Apply defense
        if agent_defending:
            boss_damage *= self.defense_reduction
        if boss_defending:
            agent_damage *= self.defense_reduction
        
        # Apply damage
        self.boss_hp -= agent_damage
        self.agent_hp -= boss_damage
        
        # Check game over
        done = self.agent_hp <= 0 or self.boss_hp <= 0
        truncated = False
        
        # Calculate reward
        reward = 0
        if done:
            if self.agent_hp <= 0:  # Agent lost
                reward = -100
            else:  # Agent won
                reward = 100 + self.agent_hp  # Bonus for remaining HP
        else:
            reward = agent_damage - boss_damage  # Reward for good trades
        
        return self._get_state(), reward, done, truncated, {}