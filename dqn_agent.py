# dqn_agent.py

from collections import deque
import random
import numpy as np
from tensorflow.keras import models, layers, optimizers


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.min_memory_size = 8
        self.steps = 0
        self.model = self._build_model()

    def _build_model(self):
        """Build the neural network for Q-value approximation."""
        model = models.Sequential()
        model.add(layers.Dense(24, activation='relu', input_shape=(self.state_size,)))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            print(f"Random action chosen (ε={self.epsilon:.3f}): {action}")
            return action

        q_values = self.model.predict(state, verbose=0)
        action = np.argmax(q_values[0])
        print(f"DQN action chosen (ε={self.epsilon:.3f}): {action}")
        print(f"Q-values: {q_values[0]}")
        return action

    def replay(self, batch_size):
        """Train on a batch of experiences."""
        if len(self.memory) < batch_size:
            print(f"Memory buffer size: {len(self.memory)}/{batch_size}")
            return

        print("\n--- Training Batch ---")
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q_values = self.model.predict(next_state, verbose=0)[0]
                target = reward + self.gamma * np.amax(next_q_values)

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            total_loss += history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(f"Loss: {total_loss / batch_size:.4f}, ε: {self.epsilon:.3f}")

