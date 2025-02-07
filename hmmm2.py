

import tkinter as tk
import random
import numpy as np
from collections import deque
from tensorflow.keras import models, layers, optimizers
import os
import json
from datetime import datetime

# Maze configuration
MAZE_WIDTH = 20
MAZE_HEIGHT = 20
TILE_SIZE = 20

# Colors for tiles
COLORS = {
    0: "white",  # Empty
    1: "black",  # Wall
    2: "blue",  # Character
    3: "green",  # +reward item
    4: "yellow",  # Start
    5: "red",  # Finish
    6: "purple"  # -reward item
}


class MazeGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RPG Maze Visualization")
        self.canvas = tk.Canvas(self.root, width=MAZE_WIDTH * TILE_SIZE, height=MAZE_HEIGHT * TILE_SIZE)
        self.canvas.pack()

        # Initialize maze and positions
        self.maze = [[1 for _ in range(MAZE_WIDTH)] for _ in range(MAZE_HEIGHT)]
        self.start = (1, 1)
        self.finish = (MAZE_WIDTH - 2, MAZE_HEIGHT - 2)
        self.character_pos = self.start

        # Generate initial maze
        self.generate_maze()
        self.add_items()

        # Initialize DQN agent
        self.state_size = MAZE_WIDTH * MAZE_HEIGHT
        self.action_size = 4
        self.agent = DQNAgent(self.state_size, self.action_size)

        # Movement settings
        self.use_dqn = True
        self.is_moving = False

        # Add control buttons
        self.add_controls()

    def add_controls(self):
        """Add control buttons to the interface"""
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        tk.Button(control_frame, text="Start DQN", command=self.start_dqn).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Start Random", command=self.start_random).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Reset", command=self.reset_game).pack(side=tk.LEFT, padx=5)

    def generate_maze(self):
        """Generate maze using DFS"""
        stack = [self.start]
        while stack:
            x, y = stack[-1]
            self.maze[y][x] = 0
            neighbors = []
            for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < MAZE_WIDTH and 0 <= ny < MAZE_HEIGHT and self.maze[ny][nx] == 1:
                    neighbors.append((nx, ny))
            if neighbors:
                nx, ny = random.choice(neighbors)
                self.maze[(y + ny) // 2][(x + nx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        self.maze[self.start[1]][self.start[0]] = 4
        self.maze[self.finish[1]][self.finish[0]] = 5

    def add_items(self, num_positive=5, num_negative=5):
        """Add reward items to the maze"""
        for _ in range(num_positive):
            while True:
                x, y = random.randint(0, MAZE_WIDTH - 1), random.randint(0, MAZE_HEIGHT - 1)
                if self.maze[y][x] == 0:
                    self.maze[y][x] = 3
                    break

        for _ in range(num_negative):
            while True:
                x, y = random.randint(0, MAZE_WIDTH - 1), random.randint(0, MAZE_HEIGHT - 1)
                if self.maze[y][x] == 0:
                    self.maze[y][x] = 6
                    break

    def draw_maze(self):
        """Draw the current state of the maze"""
        self.canvas.delete("all")
        for row in range(MAZE_HEIGHT):
            for col in range(MAZE_WIDTH):
                x1 = col * TILE_SIZE
                y1 = row * TILE_SIZE
                x2 = x1 + TILE_SIZE
                y2 = y1 + TILE_SIZE
                tile_type = self.maze[row][col]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLORS[tile_type], outline="gray")
        self.root.update()

    def get_state(self):
        """Convert current maze state to DQN input"""
        state = np.zeros((MAZE_WIDTH, MAZE_HEIGHT))
        state[self.character_pos[1]][self.character_pos[0]] = 1
        return state.flatten().reshape(1, -1)

    def is_valid_move(self, x, y):
        """Check if move is valid"""
        return 0 <= x < MAZE_WIDTH and 0 <= y < MAZE_HEIGHT and self.maze[y][x] != 1

    def move_character_dqn(self):
        """Move character using DQN agent"""
        if not self.is_moving:
            return

        print(f"\n=== DQN Movement Step {self.agent.steps} ===")
        print(f"Current position: {self.character_pos}")

        # Get current state and action
        state = self.get_state()
        action = self.agent.act(state)
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        new_x, new_y = self.character_pos[0] + dx, self.character_pos[1] + dy
        print(f"Attempting move to: ({new_x}, {new_y})")

        # Track move validity and outcome
        if self.is_valid_move(new_x, new_y):
            print("Move is valid")
            # Clear old position
            self.maze[self.character_pos[1]][self.character_pos[0]] = 0

            # Calculate reward
            reward = -0.1  # Small penalty for each move
            cell_type = self.maze[new_y][new_x]

            if cell_type == 3:  # Positive reward
                reward = 1.0
                print("Found positive reward! +1.0")
            elif cell_type == 6:  # Negative reward
                reward = -1.0
                print("Hit negative reward! -1.0")
            el
            if (new_x, new_y) == self.finish:  # Reached finish
                reward = 10.0
                print("Reached the finish! +10.0")
                self.is_moving = False
                # Save successful maze and model
                self.save_success()

            # Update position
            self.character_pos = (new_x, new_y)
            self.maze[new_y][new_x] = 2
            print(f"Move completed. Reward: {reward}")

            # Get new state and update DQN
            next_state = self.get_state()
            done = (new_x, new_y) == self.finish
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.steps += 1

            # Train more frequently with smaller batch size
            self.agent.replay(self.agent.min_memory_size)

            # Update display
            self.draw_maze()
        else:
            print("Invalid move - hit wall or boundary")
            # Penalize invalid moves
            reward = -0.5
            next_state = state  # State doesn't change
            self.agent.remember(state, action, reward, next_state, False)
            self.agent.steps += 1
            self.agent.replay(self.agent.min_memory_size)

        if self.is_moving:
            self.root.after(100, self.move_character_dqn)  # Faster updates

        if self.is_valid_move(new_x, new_y):
            # Clear old position
            self.maze[self.character_pos[1]][self.character_pos[0]] = 0

            # Calculate reward
            reward = -1  # Default step penalty
            if self.maze[new_y][new_x] == 3:  # Positive reward
                reward = 10
                print("Found positive reward!")
            elif self.maze[new_y][new_x] == 6:  # Negative reward
                reward = -10
                print("Hit negative reward!")
            elif (new_x, new_y) == self.finish:  # Reached finish
                reward = 100
                print("Reached the finish!")
                self.is_moving = False

            # Update position
            self.character_pos = (new_x, new_y)
            self.maze[new_y][new_x] = 2

            # Get new state and update DQN
            next_state = self.get_state()
            done = (new_x, new_y) == self.finish
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.replay(32)

            # Update display
            self.draw_maze()

            if self.is_moving:
                self.root.after(200, self.move_character_dqn)

    def move_character_random(self):
        """Move character randomly"""
        if not self.is_moving:
            return

        x, y = self.character_pos
        valid_moves = [(x + dx, y + dy) for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]
                       if self.is_valid_move(x + dx, y + dy)]

        if valid_moves:
            # Clear old position
            self.maze[y][x] = 0

            # Move to new position
            new_x, new_y = random.choice(valid_moves)
            self.character_pos = (new_x, new_y)

            if self.maze[new_y][new_x] == 3:
                print("Found positive reward!")
            elif self.maze[new_y][new_x] == 6:
                print("Hit negative reward!")
            elif (new_x, new_y) == self.finish:
                print("Reached the finish!")
                self.is_moving = False

            self.maze[new_y][new_x] = 2
            self.draw_maze()

            if self.is_moving:
                self.root.after(200, self.move_character_random)

    def start_dqn(self):
        """Start DQN movement"""
        self.use_dqn = True
        self.is_moving = True
        self.move_character_dqn()

    def start_random(self):
        """Start random movement"""
        self.use_dqn = False
        self.is_moving = True
        self.move_character_random()

    def save_success(self):
        """Save the successful maze and trained model"""
        # Save maze configuration
        import json
        import os
        from datetime import datetime

        # Create a directory for saves if it doesn't exist
        os.makedirs('successful_runs', exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save maze configuration
        maze_data = {
            'maze': self.maze,
            'start': self.start,
            'finish': self.finish,
            'steps': self.agent.steps
        }
        maze_file = f'successful_runs/maze_{timestamp}.json'
        with open(maze_file, 'w') as f:
            json.dump(maze_data, f)
        print(f"Maze configuration saved to {maze_file}")

        # Save trained model
        model_file = f'successful_runs/model_{timestamp}.h5'
        self.agent.model.save(model_file)
        print(f"Trained model saved to {model_file}")

        # Save training stats
        stats = {
            'total_steps': self.agent.steps,
            'final_epsilon': self.agent.epsilon,
            'memory_size': len(self.agent.memory)
        }
        stats_file = f'successful_runs/stats_{timestamp}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        print(f"Training stats saved to {stats_file}")

    def load_maze(self, maze_file):
        """Load a saved maze configuration"""
        import json
        with open(maze_file, 'r') as f:
            data = json.load(f)
        self.maze = data['maze']
        self.start = tuple(data['start'])
        self.finish = tuple(data['finish'])
        self.character_pos = self.start
        self.draw_maze()
        print(f"Loaded maze configuration from {maze_file}")

    def load_model(self, model_file):
        """Load a saved model"""
        from tensorflow.keras.models import load_model
        self.agent.model = load_model(model_file)
        print(f"Loaded trained model from {model_file}")

    def reset_game(self):
        """Reset the game state"""
        self.is_moving = False
        self.maze = [[1 for _ in range(MAZE_WIDTH)] for _ in range(MAZE_HEIGHT)]
        self.character_pos = self.start
        self.generate_maze()
        self.add_items()
        self.draw_maze()

    def run(self):
        """Start the game"""
        self.draw_maze()
        self.root.mainloop()


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.999  # Slower decay rate
        self.learning_rate = 0.001
        self.min_memory_size = 8  # Reduced from 32 to start learning earlier
        self.steps = 0  # Track total steps
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
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
            old_q = target_f[0][action]
            target_f[0][action] = target

            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            total_loss += history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(f"Loss: {total_loss / batch_size:.4f}, ε: {self.epsilon:.3f}")


if __name__ == "__main__":
    game = MazeGame()
    game.load_maze('successful_runs/maze_20250128_134712.json')
    game.load_model('successful_runs/model_20250128_134712.h5')
    game.run()